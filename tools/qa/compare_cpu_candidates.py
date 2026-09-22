"""Isolated CPU listening experiment. Never promotes outputs to the live master.

Run with venv_cpu for synthesis; use venv with --audit-only for offline ASR.
A is the unchanged runtime, B is phrase-level synthesis with unchanged voices/speed.
B corrects only Hooray stress; no noise gate, denoising, EQ, pitch shift or trimming.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cpu_voice_engine as cpu

# Each part retains all words and punctuation of normalize_text(story[index]).
# No word-only Ew fragment: avoid forcing a tiny, unstable synthesis window.
PLANS = {
    19: ["Wait, why is this tow truck hooking up my car?", "What's going on,", "stop it!"],
    30: ["we used to be friendly neighbors,", "grilling barbecue together in the backyard..."],
    3: ["Hahaha! The show is starting!", "Go Karen, destroy them with facts!"],
    2: ["Oh... I'm so sorry,", "my boyfriend and I parked there.", "We had nowhere else to park,", "your tone is really scaring me..."],
    40: ["Hooray, the bumper is intact,", "we're making it to the last screening!", "We can finally breathe out."],
    5: ["Dear residents,", "I remind you of the need to observe the rules of politeness in the neighborhood chat."],
    24: ["Ew, someone sent a photo of a dirty puddle where the car was parked.", "How gross."],
}
ORDER = list(PLANS)


def validate_plans(story, plans=PLANS):
    for index, parts in plans.items():
        if not parts or any(not p.strip() for p in parts):
            raise ValueError('Empty synthesis part')
        if cpu.normalize_text(' '.join(parts)) != cpu.normalize_text(story[index]['text']):
            raise ValueError(f'Plan changes or drops text: {index}')


def correct_hooray(phonemes):
    old, new = 'hˈɔːɹeɪ', 'həɹˈeɪ'
    if not phonemes.startswith(old + ','):
        raise ValueError('Unexpected Hooray phonemes; review before substitution')
    return new + phonemes[len(old):]


def boundary_quiet_samples(audio, leading=False):
    # Used only to AVOID adding redundant silence, never to remove samples.
    quiet = np.abs(audio) < 10 ** (-70 / 20)
    if not leading:
        quiet = quiet[::-1]
    nonquiet = np.flatnonzero(~quiet)
    return int(nonquiet[0]) if len(nonquiet) else len(audio)


def join_preserving_audio(parts, texts):
    result = [parts[0]]
    added = []
    for previous, current, text in zip(parts, parts[1:], texts):
        target_ms = 240 if text.rstrip().endswith(('.', '!', '?')) else 140
        existing = boundary_quiet_samples(previous) + boundary_quiet_samples(current, True)
        n = max(0, round(target_ms * cpu.SAMPLE_RATE / 1000) - existing)
        result.extend([np.zeros(n, np.float32), current])
        added.append(round(n * 1000 / cpu.SAMPLE_RATE, 3))
    return np.concatenate(result), added


def render_candidate(engine, item, index):
    voice = engine.cfg['characters'][item['sender_id']]
    pacing = engine.cfg['emotions'][item['emotion']]
    speed = float(np.clip(voice['speed'] * pacing['speed'], 0.5, 2))
    started = time.perf_counter()
    parts, phonetic_changes = [], []
    for text in PLANS[index]:
        synthesis_input, is_phonemes = text, False
        if text.startswith('Hooray,'):
            old = engine.tts.tokenizer.phonemize(text, engine.cfg['language'])
            synthesis_input, is_phonemes = correct_hooray(old), True
            if engine.tts.tokenizer.known(synthesis_input) != synthesis_input:
                raise ValueError('Pronunciation override includes unsupported symbols')
            phonetic_changes.append({'text': text, 'before': old, 'after': synthesis_input,
                'source': 'https://dictionary.cambridge.org/us/pronunciation/english/hooray'})
        audio, sr = engine.tts.create(synthesis_input, voice=item['sender_id'], speed=speed,
            lang=engine.cfg['language'], is_phonemes=is_phonemes, trim=False,
            sentence_pause=0.0, clause_pause=0.0)
        if sr != cpu.SAMPLE_RATE or not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Invalid synthesis output')
        parts.append(np.asarray(audio, dtype=np.float32))
    joined, pauses = join_preserving_audio(parts, PLANS[index])
    audio = cpu.finish_audio(joined)  # One gain adjustment per complete utterance, not per fragment.
    elapsed = time.perf_counter() - started
    stats = cpu.audio_stats(audio, cpu.normalize_text(item['text']))
    return audio, {'generation_seconds': round(elapsed, 4), 'speed': speed,
        'rtf': round(elapsed / stats['audio_seconds'], 4), 'segments': PLANS[index],
        'added_join_silence_ms': pauses, 'phonetic_changes': phonetic_changes,
        'gap_ms': pacing['gap_ms'], 'cache_hit': False, **stats}


def protected_files():
    paths = [cpu.ROOT / 'cpu_voice_engine.py', cpu.CONFIG, cpu.ROOT / 'story.json',
             cpu.MODEL, cpu.PROFILE_DIR / 'characters.npz', cpu.PROFILE_DIR / 'manifest.json']
    paths += sorted(p for p in cpu.OUTPUT.rglob('*') if p.is_file())
    return {str(p): cpu.file_hash(p) for p in paths}


def snapshot(process):
    memory = process.memory_info()
    return {'rss_mib': round(memory.rss / 2**20, 2),
            'process_peak_rss_mib': round(getattr(memory, 'peak_wset', memory.rss) / 2**20, 2)}


def save_collection(dest, name, records):
    parts = [np.zeros(round(cpu.SAMPLE_RATE * .25), np.float32)]
    offset = len(parts[0])
    rows = []
    for audio, row in records:
        rows.append({**row, 'start_sample': offset, 'end_sample': offset + len(audio),
                     'start_ms': round(offset * 1000 / cpu.SAMPLE_RATE),
                     'end_ms': round((offset + len(audio)) * 1000 / cpu.SAMPLE_RATE)})
        pause_ms = 600 if row['variant'] == 'A' and name == 'comparison_AB' else 900
        parts.extend([audio, np.zeros(round(cpu.SAMPLE_RATE * pause_ms / 1000), np.float32)])
        offset += len(audio) + len(parts[-1])
    output = np.concatenate(parts)
    cpu.write_wav(dest / f'{name}.wav', output)
    cpu.write_json(dest / f'{name}_timestamps.json', rows)
    saved = cpu.read_wav(dest / f'{name}.wav')
    assert len(saved) == len(output)
    previous = 0
    for (audio, _), row in zip(records, rows):
        assert previous <= row['start_sample'] < row['end_sample'] <= len(saved)
        quantized = np.rint(audio * 32768).astype('<i2').astype(np.float32) / 32768
        assert np.array_equal(saved[row['start_sample']:row['end_sample']], quantized)
        previous = row['end_sample']
    stats = cpu.audio_stats(saved, ' '.join(row['spoken_text'] for row in rows))
    return {'file': str(dest / f'{name}.wav'), 'sha256': cpu.file_hash(dest / f'{name}.wav'),
        'timestamps_sha256': cpu.file_hash(dest / f'{name}_timestamps.json'),
        'sample_exact_interval_check': True, 'row_count': len(rows), **stats}


def generate(output_dir, variant):
    import psutil
    # Different processes for A/B prevent peak-memory carryover. Output paths cannot
    # point into the live delivery; mkdir(exist_ok=False) prevents silent overwrite.
    output_dir = output_dir.resolve()
    safe_parent = (ROOT / 'output' / 'cpu_audio_candidates').resolve()
    if safe_parent not in output_dir.parents:
        raise ValueError('Output must be a NEW directory under output/cpu_audio_candidates')
    target = output_dir / variant
    if target.exists():
        raise FileExistsError(target)
    story = cpu.read_json(ROOT / 'story.json')
    validate_plans(story)
    before = protected_files()
    process = psutil.Process()
    started = time.perf_counter()
    engine = cpu.CPUVoiceEngine(threads=2)
    load_seconds = time.perf_counter() - started
    assert engine.providers == ['CPUExecutionProvider']
    target.mkdir(parents=True, exist_ok=False)
    records = []
    source_audio = cpu.read_wav(cpu.OUTPUT / 'final_voiceover_cpu.wav')
    source_rows = {r['line_index']: r for r in cpu.read_json(cpu.OUTPUT / 'final_voiceover_cpu_timestamps.json')}
    for index in ORDER:
        item = story[index]
        if variant == 'A':
            audio, meta = engine.render(item['text'], item['sender_id'], item['emotion'], cache_dir=None)
            speed = engine.cfg['characters'][item['sender_id']]['speed'] * engine.cfg['emotions'][item['emotion']]['speed']
            meta['speed'] = speed
        else:
            audio, meta = render_candidate(engine, item, index)
        baseline = source_rows[index]
        original = source_audio[round(baseline['start_ms'] * 24):round(baseline['end_ms'] * 24)]
        quantized = np.rint(audio * 32768).astype('<i2').astype(np.float32) / 32768
        row = {'line_index': index, **item, 'spoken_text': cpu.normalize_text(item['text']),
               'variant': variant, 'original_master_bit_exact_match': np.array_equal(original, quantized),
               **meta, **snapshot(process)}
        cpu.write_wav(target / f'line_{index:02d}.wav', audio)
        records.append((audio, row))
        print(f'{variant} line {index}: {meta["audio_seconds"]:.3f}s, generation {meta["generation_seconds"]:.3f}s', flush=True)
    media = save_collection(target, 'selection', records)
    after = protected_files()
    if before != after:
        raise RuntimeError('Protected project files changed during experiment')
    cpu.write_json(target / 'benchmark.json', {'variant': variant, 'status': 'requires_human_listening',
        'model_load_seconds': round(load_seconds, 3), 'generation_seconds': round(sum(r[1]['generation_seconds'] for r in records), 4),
        'speech_seconds': round(sum(r[1]['audio_seconds'] for r in records), 4),
        'providers': engine.providers, 'threads': 2, 'cache_hits': 0,
        'runtime': engine.runtime_identity, 'experiment_code_sha256': cpu.file_hash(__file__),
        'protected_files_unchanged': True, 'protected_hashes': before, 'media': media, **snapshot(process)})
    print(f'Finished {variant}; protected files unchanged.', flush=True)


def assemble(output_dir):
    output_dir = output_dir.resolve()
    if (output_dir / 'comparison_AB.wav').exists():
        raise FileExistsError(output_dir / 'comparison_AB.wav')
    variants = {v: cpu.read_json(output_dir / v / 'selection_timestamps.json') for v in ('A', 'B')}
    assert [r['line_index'] for r in variants['A']] == ORDER == [r['line_index'] for r in variants['B']]
    records = []
    for a, b in zip(variants['A'], variants['B']):
        assert a['spoken_text'] == b['spoken_text'] and a['sender_id'] == b['sender_id']
        assert a['speed'] == b['speed']
        for row in (a, b):
            audio = cpu.read_wav(output_dir / row['variant'] / f'line_{row["line_index"]:02d}.wav')
            records.append((audio, row))
    media = save_collection(output_dir, 'comparison_AB', records)
    a_bench, b_bench = [cpu.read_json(output_dir / v / 'benchmark.json') for v in ('A', 'B')]
    cpu.write_json(output_dir / 'comparison_metrics.json', {'order': ORDER,
        'all_six_characters': len({r['sender_id'] for r in variants['B']}) == 6,
        'original_spoken_words_and_punctuation_preserved': True, 'voices_and_speeds_unchanged': True,
        'trimmed_samples': 0, 'denoising': False, 'auditory_approval': False, 'media': media,
        'variants': {v: {k: b[k] for k in ('generation_seconds', 'speech_seconds', 'model_load_seconds', 'rss_mib', 'process_peak_rss_mib', 'protected_files_unchanged')}
                     for v, b in [('A', a_bench), ('B', b_bench)]}})
    print('A/B assembled; sample-exact interval checks passed.', flush=True)


def audit(output_dir):
    # An isolated audit: no edits to audit_cpu_audio.py, no writes to cpu_audio.
    from tools.qa.audit_cpu_audio import align
    from faster_whisper import WhisperModel
    import librosa
    model = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
    for variant in ('A', 'B'):
        dest = output_dir / variant
        report_path = dest / 'asr_audit.json'
        if report_path.exists():
            raise FileExistsError(report_path)
        wav, timings = dest / 'selection.wav', dest / 'selection_timestamps.json'
        sha, timing_sha = cpu.file_hash(wav), cpu.file_hash(timings)
        audio = cpu.read_wav(wav)
        rows = cpu.read_json(timings)
        results = []
        for row in rows:
            segment = audio[row['start_sample']:row['end_sample']]
            stream, _ = model.transcribe(librosa.resample(segment, orig_sr=cpu.SAMPLE_RATE, target_sr=16000),
                language='en', beam_size=3, vad_filter=False, condition_on_previous_text=False)
            transcript = ' '.join(s.text.strip() for s in stream)
            result = {'line_index': row['line_index'], 'variant': variant, 'expected': row['spoken_text'],
                      'transcript': transcript, **align(row['spoken_text'], transcript)}
            results.append(result)
            print(f'{variant} line {row["line_index"]} ASR: {transcript}', flush=True)
        assert sha == cpu.file_hash(wav) and timing_sha == cpu.file_hash(timings)
        errors, count = sum(r['word_errors'] for r in results), sum(r['expected_words'] for r in results)
        cpu.write_json(report_path, {'audio_sha256': sha, 'timestamps_sha256': timing_sha,
            'line_count': len(rows), 'word_errors': errors, 'expected_words': count,
            'wer': round(errors / count, 5), 'exact_lines': sum(r['word_errors'] == 0 for r in results),
            'note': 'ASR does not establish naturalness, emotions, noise reduction or human acceptance.', 'rows': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--variant', choices=['A', 'B'])
    actions.add_argument('--assemble', action='store_true')
    actions.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.variant:
        generate(args.output_dir, args.variant)
    elif args.assemble:
        assemble(args.output_dir)
    else:
        audit(args.output_dir)


if __name__ == '__main__':
    main()
