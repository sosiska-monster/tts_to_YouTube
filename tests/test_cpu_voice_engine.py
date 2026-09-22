import json
from pathlib import Path

import numpy as np
import pytest

import cpu_voice_engine as cpu


def test_spoken_text_preserves_unpunctuated_tail():
    assert cpu.normalize_text("Hello! Keep this last clause") == "Hello! Keep this last clause"


def test_sigh_is_not_read_as_a_word():
    assert cpu.normalize_text("Sigh, we used to grill BBQ together...") == "we used to grill barbecue together..."


def test_sigh_in_sentence_is_not_deleted():
    assert cpu.normalize_text("I heard a sigh.") == "I heard a sigh."


@pytest.mark.parametrize("text", ["", "  ", "[sigh]", None])
def test_empty_or_only_stage_direction_rejected(text):
    with pytest.raises(ValueError):
        cpu.normalize_text(text)


def test_finish_audio_bounds_and_preserves_input():
    x = np.sin(np.arange(24000) * 0.07).astype(np.float32) * 1.4
    original = x.copy()
    y = cpu.finish_audio(x)
    assert np.array_equal(x, original)
    assert np.max(np.abs(y)) <= 10**(-1/20) + 1e-6
    assert y[0] == y[-1] == 0
    assert len(y) == len(x)


@pytest.mark.parametrize("x", [np.array([]), np.zeros(1000), np.array([np.nan, 1])])
def test_invalid_audio_rejected(x):
    with pytest.raises(ValueError):
        cpu.finish_audio(x)


def test_pcm_roundtrip(tmp_path):
    x = np.linspace(-0.8, 0.8, 1200).astype(np.float32)
    path = tmp_path / "test.wav"
    cpu.write_wav(path, x)
    assert np.max(np.abs(cpu.read_wav(path) - x)) <= 1 / 32768


def test_fingerprint_sensitive_to_text_voice_model_and_speed():
    original = {"text": "Hello.", "voice": "a", "model": "one", "speed": 1.0}
    for key, value in [("text", "Goodbye."), ("voice", "b"), ("model", "two"), ("speed", 0.9)]:
        assert cpu.content_hash(original) != cpu.content_hash({**original, key: value})
    assert cpu.content_hash(original) == cpu.content_hash(dict(reversed(list(original.items()))))


def test_runtime_cache_checks_wav_content_and_parameters(tmp_path):
    class Stub:
        calls = 0
        def create(self, *args, **kwargs):
            self.calls += 1
            return (np.sin(np.arange(24000) * 0.1) * 0.1).astype(np.float32), 24000
    engine = cpu.CPUVoiceEngine.__new__(cpu.CPUVoiceEngine)
    engine.tts = Stub()
    engine.cfg = {"characters": {"a": {"voice": "source", "speed": 1.0}},
                  "emotions": {"neutral": {"speed": 1.0, "gap_ms": 300}}, "language": "en-us"}
    engine.runtime_identity = {"model": "one", "voices": "one"}
    first, m1 = engine.render("Hello there.", "a", cache_dir=tmp_path)
    second, m2 = engine.render("Hello there.", "a", cache_dir=tmp_path)
    assert engine.tts.calls == 1 and m2["cache_hit"]
    assert np.array_equal(first, second)
    cpu.write_wav(tmp_path / f"{m1['key']}.wav", first * 0.5)
    _, m3 = engine.render("Hello there.", "a", cache_dir=tmp_path)
    assert engine.tts.calls == 2 and not m3["cache_hit"]
    engine.cfg["characters"]["a"]["speed"] = 0.9
    _, m4 = engine.render("Hello there.", "a", cache_dir=tmp_path)
    assert engine.tts.calls == 3 and m4["key"] != m1["key"]
    engine.runtime_identity["model"] = "two"
    _, m5 = engine.render("Hello there.", "a", cache_dir=tmp_path)
    assert engine.tts.calls == 4 and m5["key"] != m4["key"]
    with pytest.raises(ValueError):
        engine.render("Hello.", "missing", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        engine.render("Hello.", "a", emotion="missing", cache_dir=tmp_path)


def test_bake_exports_only_selected_matrices(tmp_path):
    model = tmp_path / "test.onnx"
    model.write_bytes(b"test-model-not-for-inference")
    source = tmp_path / "source.npz"
    np.savez(source, keep=np.ones((510, 1, 256), np.float32), discard=np.zeros((510, 1, 256), np.float32))
    config = tmp_path / "config.json"
    cpu.write_json(config, {"characters": {"actor": {"voice": "keep", "speed": 1.0}},
                            "emotions": {"neutral": {"speed": 1.0, "gap_ms": 300}}})
    manifest = cpu.bake_profiles(config, source, tmp_path / "profiles", model)
    with np.load(tmp_path / "profiles" / "characters.npz", allow_pickle=False) as result:
        assert result.files == ["actor"]
        assert result["actor"].shape == (510, 1, 256)
    assert manifest["model_required"] is True
    assert manifest["model_sha256"] == cpu.file_hash(model)


def test_current_config_covers_entire_story():
    cfg = cpu.validate_config(cpu.read_json(cpu.CONFIG))
    story = cpu.read_json(cpu.ROOT / "story.json")
    assert len(cfg["characters"]) == 6
    assert len(cfg["emotions"]) == 7
    assert all(row["sender_id"] in cfg["characters"] and row["emotion"] in cfg["emotions"] for row in story)
    assert len({(row["sender_id"], row["emotion"]) for row in story}) == 42


@pytest.fixture
def delivery_inputs(tmp_path):
    master = tmp_path / "master.wav"
    timings = tmp_path / "timestamps.json"
    cpu.write_wav(master, np.sin(np.arange(24000) * 0.1).astype(np.float32) * 0.1)
    story = [{"sender_id": "actor", "emotion": "neutral", "text": "Hello."}]
    cpu.write_json(timings, [{"line_index": 0, **story[0], "start_ms": 0, "end_ms": 1000}])
    benchmark = {"master_sha256": cpu.file_hash(master)}
    audit = {"audio_sha256": cpu.file_hash(master), "timestamps_sha256": cpu.file_hash(timings),
             "line_count": 1, "rows": [{"line_index": 0, "expected": "Hello.",
                                        "character": "actor", "emotion": "neutral"}]}
    return master, timings, benchmark, audit, story


def test_delivery_accepts_bound_audit(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    audio, rows = validate_delivery(*delivery_inputs)
    assert len(audio) == 24000 and len(rows) == 1


@pytest.mark.parametrize("field", ["audio_sha256", "timestamps_sha256"])
def test_delivery_rejects_missing_audit_fingerprint(delivery_inputs, field):
    from tools.package_cpu_audio import validate_delivery
    delivery_inputs[3].pop(field)
    with pytest.raises(ValueError, match="checksum"):
        validate_delivery(*delivery_inputs)


def test_delivery_rejects_changed_audio(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    master = delivery_inputs[0]
    cpu.write_wav(master, cpu.read_wav(master) * 0.5)
    with pytest.raises(ValueError, match="Benchmark"):
        validate_delivery(*delivery_inputs)


def test_delivery_rejects_changed_story(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    delivery_inputs[4][0]["text"] = "Goodbye."
    with pytest.raises(ValueError, match="current story"):
        validate_delivery(*delivery_inputs)


def test_delivery_rejects_incomplete_audit(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    delivery_inputs[3]["rows"] = []
    with pytest.raises(ValueError, match="Incomplete"):
        validate_delivery(*delivery_inputs)


def test_delivery_rejects_invalid_intervals(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    timings = delivery_inputs[1]
    rows = cpu.read_json(timings)
    rows[0]["end_ms"] = 2000
    cpu.write_json(timings, rows)
    delivery_inputs[3]["timestamps_sha256"] = cpu.file_hash(timings)
    with pytest.raises(ValueError, match="intervals"):
        validate_delivery(*delivery_inputs)


def test_delivery_rejects_wrong_asr_text(delivery_inputs):
    from tools.package_cpu_audio import validate_delivery
    delivery_inputs[3]["rows"][0]["expected"] = "Goodbye."
    with pytest.raises(ValueError, match="ASR row"):
        validate_delivery(*delivery_inputs)


def test_candidate_plans_preserve_every_word_and_all_characters():
    from tools.qa.compare_cpu_candidates import validate_plans, PLANS
    story = cpu.read_json(cpu.ROOT / "story.json")
    validate_plans(story)
    assert {story[i]["sender_id"] for i in PLANS} == set(cpu.read_json(cpu.CONFIG)["characters"])


def test_candidate_rejects_lost_tail():
    from tools.qa.compare_cpu_candidates import validate_plans
    with pytest.raises(ValueError, match="drops text"):
        validate_plans([{"text": "Hello! Keep the end."}], {0: ["Hello!"]})


def test_hooray_override_only_changes_leading_word():
    from tools.qa.compare_cpu_candidates import correct_hooray
    suffix = ", ðə bˈʌmpɚɹ ɪz ɪntˈækt,"
    assert correct_hooray("hˈɔːɹeɪ" + suffix) == "həɹˈeɪ" + suffix


def test_hooray_override_rejects_unexpected_phonemes():
    from tools.qa.compare_cpu_candidates import correct_hooray
    with pytest.raises(ValueError, match="Unexpected"):
        correct_hooray("different phonemes")


def test_phrase_join_preserves_all_samples_and_tail():
    from tools.qa.compare_cpu_candidates import join_preserving_audio
    first, last = np.full(100, .2, np.float32), np.full(200, -.1, np.float32)
    joined, gaps = join_preserving_audio([first, last], ["Hello,", "keep the end."])
    assert np.array_equal(joined[:100], first)
    assert np.array_equal(joined[-200:], last)
    assert np.all(joined[100:-200] == 0)
    assert gaps == [140.0]


def test_phrase_join_never_trims_existing_long_silence():
    from tools.qa.compare_cpu_candidates import join_preserving_audio
    first = np.concatenate([np.full(100, .2, np.float32), np.zeros(8000, np.float32)])
    last = np.concatenate([np.zeros(8000, np.float32), np.full(100, .1, np.float32)])
    joined, gaps = join_preserving_audio([first, last], ["Hello!", "Keep this."])
    assert np.array_equal(joined, np.concatenate([first, last]))
    assert gaps == [0.0]


def test_candidate_refuses_live_delivery_before_initialization():
    from tools.qa.compare_cpu_candidates import generate
    with pytest.raises(ValueError, match="NEW directory"):
        generate(cpu.OUTPUT, "B")


@pytest.mark.parametrize('text, expected', [
    ('Hahaha! Keep the end', 'Ha Ha Ha! Keep the end'),
    ('Sight is good.', 'Sight is good.'),
    ('Sighing is normal.', 'Sighing is normal.'),
    ('Dr. Smith paid $12.50 at 5:05 PM.',
     'Doctor Smith paid twelve dollars and fifty cents at five oh five P M.'),
    ('CPU, API, USB, AI, HTTPS.', 'C P U, A P I, U S B, A I, H T T P S.'),
    ('Karen, wait for me.', 'Karen, wait for me.'),
    ('It is 25% and 5 kg.', 'It is 25 percent and 5 kilograms.'),
])
def test_extended_normalization(text, expected):
    assert cpu.normalize_text(text) == expected


@pytest.mark.parametrize('text', ['[laughs] Hello.', '[gasp]', '\u041f\u0440\u0438\u0432\u0435\u0442', 'Hello\x01'])
def test_unsupported_inputs_fail_closed(text):
    with pytest.raises(ValueError):
        cpu.normalize_text(text)


class FakeTokenizer:
    def __init__(self, phonemes):
        self.phonemes = phonemes
    def phonemize(self, text, language):
        return self.phonemes
    def known(self, value):
        return value


def fake_tts(phonemes):
    from types import SimpleNamespace
    return SimpleNamespace(tokenizer=FakeTokenizer(phonemes))


def test_hooray_preserves_whole_utterance():
    text, phonetic, fixes = cpu.pronunciation_input(
        fake_tts('hˈɔːɹeɪ, wiː mˈeɪd ɪt!'), 'Hooray, we made it!', 'en-us')
    assert text == 'həɹˈeɪ, wiː mˈeɪd ɪt!' and phonetic
    assert fixes[0]['count'] == 1


def test_ew_not_enabled_by_default():
    text, phonetic, fixes = cpu.pronunciation_input(object(), 'Ew, awful.', 'en-us')
    assert (text, phonetic, fixes) == ('Ew, awful.', False, [])


def test_ew_ambiguity_does_not_change_you():
    original = 'jˈuː, jˈuː!'
    text, phonetic, fixes = cpu.pronunciation_input(fake_tts(original), 'Ew, you!', 'en-us', include_ew=True)
    assert text == 'Ew, you!' and not phonetic
    assert fixes[0]['status'].startswith('not_applied')


def test_unsupported_override_rejected():
    tts = fake_tts('hˈɔːɹeɪ!')
    tts.tokenizer.known = lambda value: ''
    with pytest.raises(ValueError, match='unsupported symbols'):
        cpu.pronunciation_input(tts, 'Hooray!', 'en-us')


def test_trim_preserves_interior_and_safety_guards():
    x = np.zeros(30000, np.float32)
    x[8000:9000] = .1
    x[18000:19000] = .2
    original = x.copy()
    y, removed = cpu.trim_outer_silence(x)
    start, end = 8000 - 1920, 19000 + 3360
    assert removed == {'head_samples': start, 'tail_samples': len(x) - end}
    assert np.array_equal(y, x[start:end]) and np.array_equal(x, original)
    assert np.all(y[9000-start:18000-start] == 0)


@pytest.mark.parametrize('kwargs', [{'head_ms': -1}, {'tail_ms': float('nan')}])
def test_trim_invalid_guards(kwargs):
    with pytest.raises(ValueError):
        cpu.trim_outer_silence(np.ones(100), **kwargs)


def test_candidate_flags_are_opt_in_and_cache_is_separate(tmp_path):
    class Stub:
        tokenizer = FakeTokenizer('hˈɔːɹeɪ!')
        calls = []
        def create(self, text, **kwargs):
            self.calls.append((text, kwargs))
            signal = np.sin(np.arange(4000) * .1).astype(np.float32) * .1
            return np.concatenate([np.zeros(8000, np.float32), signal, np.zeros(8000, np.float32)]), 24000
    engine = cpu.CPUVoiceEngine.__new__(cpu.CPUVoiceEngine)
    engine.tts = Stub()
    engine.cfg = {'characters': {'a': {'voice': 'source', 'speed': 1.0}},
                  'emotions': {'neutral': {'speed': 1.0, 'gap_ms': 300}}, 'language': 'en-us'}
    engine.runtime_identity = {'test': True}
    a, ma = engine.render('Hooray!', 'a', cache_dir=tmp_path)
    c, mc = engine.render('Hooray!', 'a', cache_dir=tmp_path, pronunciation_fixes=True, trim_edges=True)
    repeated, mr = engine.render('Hooray!', 'a', cache_dir=tmp_path, pronunciation_fixes=True, trim_edges=True)
    assert len(engine.tts.calls) == 2 and mr['cache_hit']
    assert np.array_equal(c, repeated) and len(c) < len(a)
    assert ma['key'] != mc['key'] and ma['pronunciation_fixes'] == []
    assert engine.tts.calls[0][0] == 'Hooray!' and not engine.tts.calls[0][1]['is_phonemes']
    assert engine.tts.calls[1][0] == 'həɹˈeɪ!' and engine.tts.calls[1][1]['is_phonemes']
    assert mc['pronunciation_fixes'][0]['word'] == 'Hooray'
    with pytest.raises(ValueError, match='requires'):
        engine.render('Ew.', 'a', experimental_ew=True)


def test_stream_matches_pcm_stats_without_whole_story_buffer(tmp_path):
    parts = [np.zeros(20, np.float32), np.linspace(-.8, .7, 12000, dtype=np.float32)]
    path = tmp_path / 'stream.wav'
    with cpu.WavStream(path) as stream:
        for part in parts:
            stream.append(part)
    recorded = cpu.read_wav(path)
    assert len(recorded) == stream.frames == sum(map(len, parts))
    assert stream.stats('Hello world') == cpu.audio_stats(recorded, 'Hello world')


def test_stream_failure_keeps_existing_master(tmp_path):
    path = tmp_path / 'master.wav'
    path.write_bytes(b'previous-master')
    with pytest.raises(ValueError, match='Invalid stream'):
        with cpu.WavStream(path) as stream:
            stream.append(np.ones(10, np.float32) * .1)
            stream.append(np.array([np.nan]))
    assert path.read_bytes() == b'previous-master'


def test_stream_empty_stats_and_riff_limit(tmp_path):
    with cpu.WavStream(tmp_path / 'empty.wav') as stream:
        with pytest.raises(ValueError, match='empty'):
            stream.stats('')
        stream.frames = (0xffffffff - 36) // 2
        with pytest.raises(ValueError, match='RIFF'):
            stream.append(np.ones(2, np.float32))
        stream.frames = 0


@pytest.mark.parametrize('mode, basename, timing', [
    (['--text', 'Hello.'], 'single', 'single_timestamps.json'),
    (['--preview'], 'preview', 'preview_timestamps.json'),
    (['--lines', '0'], 'selection', 'selection_timestamps.json'),
    ([], 'final_voiceover', 'dialog_timestamps.json'),
])
@pytest.mark.parametrize('chunk_limit', [0, 160])
def test_cli_stream_report_and_separate_timestamps(tmp_path, monkeypatch, mode, basename, timing, chunk_limit):
    class StubEngine:
        providers = ['CPUExecutionProvider']
        def __init__(self, **kwargs):
            assert kwargs['low_memory'] is True
        def render(self, text, char, emotion, **kwargs):
            assert not kwargs['warmth'] and not kwargs['breath']
            assert kwargs['max_chunk_chars'] == chunk_limit
            signal = np.sin(np.arange(2400) * .1).astype(np.float32) * .1
            return signal, {'audio_seconds': .1, 'generation_seconds': .02, 'gap_ms': 300,
                            'spoken_text': text, 'rtf': .2, 'cache_hit': False}
    monkeypatch.setattr(cpu, 'CPUVoiceEngine', StubEngine)
    previous = tmp_path / 'dialog_timestamps.json'
    previous.write_text('previous-master-timings')
    monkeypatch.setattr('sys.argv', ['cpu_voice_engine', '--output-dir', str(tmp_path),
                                     '--max-chunk-chars', str(chunk_limit), *mode])
    cpu.main()
    report = cpu.read_json(tmp_path / f'{basename}_benchmark.json')
    assert report['timestamps_json'] == str(tmp_path / timing)
    rows = cpu.read_json(tmp_path / timing)
    audio = cpu.read_wav(tmp_path / f'{basename}.wav')
    assert report['line_count'] == len(rows)
    assert report['master_seconds'] == round(len(audio) / cpu.SAMPLE_RATE, 3)
    assert report['audio_stats'] == cpu.audio_stats(audio, ' '.join(r['spoken_text'] for r in rows))
    assert report['master_sha256'] == cpu.file_hash(tmp_path / f'{basename}.wav')
    assert all(a['end_ms'] < b['start_ms'] for a, b in zip(rows, rows[1:]))
    if mode:
        assert previous.read_text() == 'previous-master-timings'


def test_cli_failed_render_keeps_master_and_timings(tmp_path, monkeypatch):
    class FailingEngine:
        def __init__(self, **kwargs):
            pass
        def render(self, *args, **kwargs):
            raise RuntimeError('synthesis failure')
    monkeypatch.setattr(cpu, 'CPUVoiceEngine', FailingEngine)
    master, times = tmp_path / 'final_voiceover.wav', tmp_path / 'dialog_timestamps.json'
    master.write_bytes(b'previous-master')
    times.write_text('previous-timings')
    monkeypatch.setattr('sys.argv', ['cpu_voice_engine', '--output-dir', str(tmp_path)])
    with pytest.raises(RuntimeError, match='synthesis failure'):
        cpu.main()
    assert master.read_bytes() == b'previous-master'
    assert times.read_text() == 'previous-timings'


@pytest.mark.parametrize('blend', [{}, {'a': -1, 'b': 2}, {'a': .5}, {'a': float('nan')}, {'a': float('inf')}, {'a': '1'}])
def test_bad_blends_rejected(blend):
    cfg = {'characters': {'a': {'voice_blend': blend, 'speed': 1}},
           'emotions': {'neutral': {'speed': 1, 'gap_ms': 300}}}
    with pytest.raises(ValueError):
        cpu.validate_config(cfg)


def test_bake_weighted_blend_matches_source(tmp_path):
    model, source, config = tmp_path / 'model.onnx', tmp_path / 'source.npz', tmp_path / 'config.json'
    model.write_bytes(b'test')
    np.savez(source, a=np.ones((510, 1, 256), np.float32), b=np.zeros((510, 1, 256), np.float32))
    cpu.write_json(config, {'characters': {'actor': {'voice_blend': {'a': .6, 'b': .4}, 'speed': 1}},
                           'emotions': {'neutral': {'speed': 1, 'gap_ms': 300}}})
    cpu.bake_profiles(config, source, tmp_path / 'profiles', model)
    with np.load(tmp_path / 'profiles' / 'characters.npz') as profiles:
        assert np.allclose(profiles['actor'], .6)


def test_phoneme_fallback_keeps_orthography():
    text, is_phonemes, fixes = cpu.pronunciation_input(fake_tts('unexpected'), 'Hooray! Keep the end.', 'en-us')
    assert text == 'Hooray! Keep the end.' and not is_phonemes and not fixes


@pytest.mark.parametrize('bad_cache', ['json', 'metadata', 'empty_wav'])
def test_corrupt_cache_regenerated(tmp_path, bad_cache):
    class Stub:
        calls = 0
        def create(self, *args, **kwargs):
            self.calls += 1
            return np.sin(np.arange(2400) * .1).astype(np.float32) * .1, 24000
    engine = cpu.CPUVoiceEngine.__new__(cpu.CPUVoiceEngine)
    engine.tts = Stub()
    engine.cfg = {'characters': {'a': {'voice': 'source', 'speed': 1}},
                  'emotions': {'neutral': {'speed': 1, 'gap_ms': 300}}, 'language': 'en-us'}
    engine.runtime_identity = {'test': True}
    _, meta = engine.render('Hello.', 'a', cache_dir=tmp_path)
    wav, info = tmp_path / f"{meta['key']}.wav", tmp_path / f"{meta['key']}.json"
    if bad_cache == 'json':
        info.write_text('{broken')
    elif bad_cache == 'metadata':
        cpu.write_json(info, [])
    else:
        import wave
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        meta['wav_sha256'] = cpu.file_hash(wav)
        cpu.write_json(info, meta)
    audio, result = engine.render('Hello.', 'a', cache_dir=tmp_path)
    assert len(audio) and not result['cache_hit'] and engine.tts.calls == 2


def test_cli_does_not_reject_text_expanded_past_input_limit(tmp_path, monkeypatch):
    raw = 'CPU ' * 400
    assert len(cpu.normalize_text(raw)) > 2000
    class StubEngine:
        providers = ['CPUExecutionProvider']
        def __init__(self, **kwargs):
            pass
        def render(self, text, *args, **kwargs):
            assert text == raw
            clean = cpu.normalize_text(text)
            return np.full(2400, .1, np.float32), {
                'audio_seconds': .1, 'generation_seconds': .01, 'gap_ms': 0,
                'spoken_text': clean, 'rtf': .1, 'cache_hit': False}
    monkeypatch.setattr(cpu, 'CPUVoiceEngine', StubEngine)
    monkeypatch.setattr('sys.argv', ['cpu_voice_engine', '--text', raw, '--output-dir', str(tmp_path)])
    cpu.main()


def test_concurrent_calls_on_one_engine_are_serialized():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    class Stub:
        active = 0
        maximum = 0
        def create(self, *args, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            time.sleep(.01)
            self.active -= 1
            return np.sin(np.arange(2400) * .1).astype(np.float32) * .1, 24000
    engine = cpu.CPUVoiceEngine.__new__(cpu.CPUVoiceEngine)
    engine._render_lock = threading.Lock()
    engine.tts = Stub()
    engine.cfg = {'characters': {'a': {'voice': 'source', 'speed': 1}},
                  'emotions': {'neutral': {'speed': 1, 'gap_ms': 300}}, 'language': 'en-us'}
    engine.runtime_identity = {'test': True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: engine.render('Hello.', 'a'), range(4)))
    assert len(results) == 4 and engine.tts.maximum == 1


@pytest.fixture
def chunk_engine():
    class Stub:
        def __init__(self):
            self.calls = []
            self.parts = []
        def create(self, text, **kwargs):
            self.calls.append((text, kwargs))
            signal = np.sin(np.arange(800) * .1).astype(np.float32) * .1
            part = np.concatenate([np.zeros(4000, np.float32), signal,
                                   np.zeros(5000, np.float32)])
            self.parts.append(part.copy())
            return part, cpu.SAMPLE_RATE
    engine = cpu.CPUVoiceEngine.__new__(cpu.CPUVoiceEngine)
    engine.tts = Stub()
    engine.cfg = {'characters': {'a': {'voice': 'source', 'speed': 1}},
                  'emotions': {'neutral': {'speed': 1, 'gap_ms': 300}}, 'language': 'en-us'}
    engine.runtime_identity = {'test': True}
    return engine


@pytest.mark.parametrize('trim', [False, True])
def test_chunk_join_preserves_interior_samples_and_raw_offsets(chunk_engine, trim):
    text = ('Please keep every word in this sentence ' * 5) + 'this is the ending'
    audio, meta = chunk_engine.render(text, 'a', max_chunk_chars=80, trim_edges=trim)
    parts = chunk_engine.tts.parts
    raw = np.concatenate(parts)
    reference, removed = cpu.trim_outer_silence(raw) if trim else (raw, {'head_samples': 0, 'tail_samples': 0})
    assert np.array_equal(audio, cpu.finish_audio(reference))
    assert meta['edge_trim'] == removed
    assert meta['chunk_count'] == len(parts) > 1
    assert ' '.join(row['text'] for row in meta['synthesis_chunks']) == text
    offset = 0
    for row, part in zip(meta['synthesis_chunks'], parts):
        assert row['start_sample'] == offset
        offset += len(part)
        assert row['end_sample'] == offset
    assert meta['chunk_sample_coordinates'] == 'raw_concatenated_before_outer_trim_and_effects'
    assert all(not kwargs['trim'] and kwargs['sentence_pause'] == kwargs['clause_pause'] == 0
               for _, kwargs in chunk_engine.tts.calls)


def test_chunk_mode_is_opt_in_and_cache_keys_are_separate(chunk_engine, tmp_path):
    text = 'Please preserve every word and the last clause ' * 4
    _, legacy = chunk_engine.render(text, 'a', cache_dir=tmp_path)
    assert len(chunk_engine.tts.calls) == legacy['chunk_count'] == 1
    audio, split = chunk_engine.render(text, 'a', cache_dir=tmp_path, max_chunk_chars=80)
    calls = len(chunk_engine.tts.calls)
    again, cached = chunk_engine.render(text, 'a', cache_dir=tmp_path, max_chunk_chars=80)
    assert cached['cache_hit'] and np.array_equal(audio, again)
    assert len(chunk_engine.tts.calls) == calls
    assert legacy['key'] != split['key'] and split['chunk_count'] > 1


def test_chunk_pronunciation_uses_each_complete_chunk(chunk_engine, monkeypatch):
    seen = []
    def override(tts, text, language, **kwargs):
        seen.append(text)
        return text, True, [{'word': 'test'}]
    monkeypatch.setattr(cpu, 'pronunciation_override', override)
    text = 'Please preserve every word and the last clause ' * 4
    _, meta = chunk_engine.render(text, 'a', max_chunk_chars=80, pronunciation_fixes=True)
    assert seen == [row['text'] for row in meta['synthesis_chunks']]
    assert [text for text, _ in chunk_engine.tts.calls] == seen
    assert all(kwargs['is_phonemes'] for _, kwargs in chunk_engine.tts.calls)
    assert [fix['chunk_index'] for fix in meta['pronunciation_fixes']] == list(range(len(seen)))


@pytest.mark.parametrize('bad, rate', [(np.zeros(10), 24000), (np.array([]), 24000),
                                      (np.array([np.nan]), 24000), (np.ones((2, 2)), 24000),
                                      (np.ones(10) * .1, 16000)])
def test_bad_later_chunk_does_not_write_cache(chunk_engine, tmp_path, bad, rate):
    original = chunk_engine.tts.create
    def create(text, **kwargs):
        if chunk_engine.tts.calls:
            return bad, rate
        return original(text, **kwargs)
    chunk_engine.tts.create = create
    with pytest.raises(ValueError):
        chunk_engine.render('Please preserve every word and the last clause ' * 4,
                            'a', cache_dir=tmp_path, max_chunk_chars=80)
    assert not list(tmp_path.iterdir())


def test_chunk_expanded_text_is_not_normalized_twice(chunk_engine):
    raw = 'CPU ' * 400
    _, meta = chunk_engine.render(raw, 'a', max_chunk_chars=160)
    assert ' '.join(row['text'] for row in meta['synthesis_chunks']) == cpu.normalize_text(raw)
    assert len(meta['spoken_text']) > 2000


@pytest.mark.parametrize('limit', [-1, 79, 2001, True, 80.0])
def test_invalid_chunk_limit_fails_before_synthesis(chunk_engine, limit):
    with pytest.raises(ValueError, match='max_chunk_chars'):
        chunk_engine.render('Hello.', 'a', max_chunk_chars=limit)
    assert not chunk_engine.tts.calls


@pytest.mark.parametrize('limit', ['-1', '79', '2001', 'abc'])
def test_cli_rejects_chunk_limit_before_loading_engine(monkeypatch, limit):
    def forbidden(**kwargs):
        pytest.fail('Invalid CLI argument must not load the model')
    monkeypatch.setattr(cpu, 'CPUVoiceEngine', forbidden)
    monkeypatch.setattr('sys.argv', ['cpu_voice_engine', '--text', 'Hello.', '--max-chunk-chars', limit])
    with pytest.raises(SystemExit) as error:
        cpu.main()
    assert error.value.code == 2
