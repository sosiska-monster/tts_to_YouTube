"""Bounded offline CPU edge-case synthesis; does not rate naturalness or emotion."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cpu_voice_engine as cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a NEW output directory to preserve previous results')
    output.mkdir(parents=True)
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:1])
    engine = cpu.CPUVoiceEngine(threads=1, low_memory=True)
    cases = [
        ('short', 'Hi.', 'girl_narrator', 'neutral'),
        ('hooray', 'Hooray, we made it! Keep this final clause', 'boyfriend', 'joy'),
        ('ew_you', 'Ew, you left this here? Keep the final words.', 'karen', 'disgust'),
        ('laughter_words', 'Hahaha! The show is on. Do not lose this ending', 'toxic_guy', 'joy'),
        ('money_time', 'Dr. Smith paid $12.50 at 5:05 PM. CPU, API and USB.', 'authority', 'neutral'),
        ('quiet_text', 'Oh... I am sorry. Please do not go.', 'submissive_girl', 'sadness'),
        ('long_no_punctuation', ('Please keep every word in this sentence ' * 60)[:1900] + ' this is the ending', 'girl_narrator', 'neutral'),
    ]
    rows = []
    total_start = time.perf_counter()
    with cpu.WavStream(output / 'edge_cases.wav') as master:
        for name, text, character, emotion in cases:
            start = master.frames
            audio, meta = engine.render(text, character, emotion, cache_dir=output / 'cache',
                                        pronunciation_fixes=True, trim_edges=True)
            assert np.isfinite(audio).all() and len(audio)
            assert float(np.max(np.abs(audio))) <= 10 ** (-1 / 20) + 1 / 32768
            assert np.sqrt(np.mean(audio ** 2)) > 1e-5
            repeated, cached = engine.render(text, character, emotion, cache_dir=output / 'cache',
                                            pronunciation_fixes=True, trim_edges=True)
            assert cached['cache_hit'] and np.array_equal(audio, repeated)
            master.append(audio)
            end = master.frames
            master.append(np.zeros(cpu.SAMPLE_RATE // 2, np.float32))
            rows.append({'case': name, 'text': text, 'character': character, 'emotion': emotion,
                         'start_ms': round(start * 1000 / cpu.SAMPLE_RATE),
                         'end_ms': round(end * 1000 / cpu.SAMPLE_RATE),
                         'cache_identical': True, **meta})
            print(f"{name}: {len(audio)/cpu.SAMPLE_RATE:.2f}s, cache exact, no clipping", flush=True)
    rejected = []
    for text in ['', '[gasp]', '\u041f\u0440\u0438\u0432\u0435\u0442', 'Hello\u200b', 'a' * 2001, 'Meet at 13:00 PM.', '$12.345']:
        try:
            engine.render(text, 'girl_narrator')
        except ValueError as exc:
            rejected.append({'input': text[:60], 'input_length': len(text), 'reason': str(exc)})
        else:
            raise AssertionError('Unsupported input was accepted')
    memory = process.memory_info()
    report = {'status': 'technical_checks_passed_listening_required',
              'limitations': 'No human listening, ASR, speaker similarity or emotion accuracy test. No training or cloning.',
              'threads': 1, 'cpu_affinity': process.cpu_affinity(),
              'rss_mib': round(memory.rss / 2**20, 2),
              'peak_rss_mib': round(getattr(memory, 'peak_wset', memory.rss) / 2**20, 2),
              'wall_seconds': round(time.perf_counter() - total_start, 3),
              'rendered_cases': len(rows), 'rejected_cases': rejected,
              'master_sha256': cpu.file_hash(output / 'edge_cases.wav'),
              'master_stats': master.stats(' '.join(row['spoken_text'] for row in rows)),
              'runtime_identity': engine.runtime_identity, 'rows': rows}
    cpu.write_json(output / 'edge_report.json', report)
    print({k: v for k, v in report.items() if k not in {'rows', 'runtime_identity', 'rejected_cases'}}, flush=True)


if __name__ == '__main__':
    main()
