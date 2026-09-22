"""Offline ASR check of CPU output; ASR is not a production dependency."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cpu_voice_engine import normalize_text, file_hash, write_json


def words(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"\bbbq\b", "barbecue", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text)


def align(expected, actual):
    a, b = words(expected), words(actual)
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] != b[j-1]))
    i, j = len(a), len(b)
    operations = []
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]:
                operations.append({"kind": "substitution", "expected": a[i-1], "asr": b[j-1]})
            i -= 1
            j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            operations.append({"kind": "deletion", "expected": a[i-1]})
            i -= 1
        else:
            operations.append({"kind": "insertion", "asr": b[j-1]})
            j -= 1
    return {"word_errors": d[-1][-1], "expected_words": len(a),
            "wer": round(d[-1][-1] / max(1, len(a)), 5), "operations": list(reversed(operations))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-preview", action="store_true")
    parser.add_argument("--wav", type=Path, help="Explicit candidate WAV; never modify the source")
    parser.add_argument("--timestamps", type=Path, help="Matching candidate timestamps")
    parser.add_argument("--output-dir", type=Path, default=ROOT / 'output' / 'cpu_audio')
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if bool(args.wav) != bool(args.timestamps) or (args.baseline_preview and args.wav):
        parser.error('Supply --wav and --timestamps together, without --baseline-preview')
    if args.threads < 1:
        parser.error('--threads must be positive')
    dest = args.output_dir
    start = time.perf_counter()
    model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=args.threads, local_files_only=True)
    if args.wav:
        wav_path, timing_path = args.wav, args.timestamps
        rows = json.loads(timing_path.read_text(encoding='utf-8'))
        if isinstance(rows, dict):
            rows = [{**r, 'line_index': i, 'sender_id': r['character']} for i, r in enumerate(rows['rows'])]
        name = 'candidate_asr_audit'
    elif args.baseline_preview:
        timing_path = ROOT / "output" / "dialog_timestamps.json"
        wav_path = ROOT / "output" / "final_voiceover.wav"
        all_rows = json.loads(timing_path.read_text(encoding="utf-8"))
        rows = [{"line_index": i, **all_rows[i]} for i in [0, 1, 2, 3, 4, 5, 19, 30]]
        name = "baseline_preview_asr"
    else:
        wav_path = dest / "final_voiceover_cpu.wav"
        timing_path = dest / "final_voiceover_cpu_timestamps.json"
        rows = json.loads(timing_path.read_text(encoding="utf-8"))
        name = "cpu_asr_audit"
    audio_sha = file_hash(wav_path)
    timings_sha = file_hash(timing_path)
    audio, sr = sf.read(wav_path, dtype="float32")
    results = []
    for row in rows:
        segment = audio[round(row["start_ms"] * sr / 1000):round(row["end_ms"] * sr / 1000)]
        asr_audio = librosa.resample(segment, orig_sr=sr, target_sr=16000)
        segments, _ = model.transcribe(asr_audio, language="en", beam_size=3, vad_filter=False,
                                        condition_on_previous_text=False)
        transcript = " ".join(s.text.strip() for s in segments)
        expected = normalize_text(row["text"])
        result = {"line_index": row["line_index"], "character": row["sender_id"],
                  "emotion": row["emotion"], "expected": expected, "transcript": transcript,
                  **align(expected, transcript)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    errors = sum(r["word_errors"] for r in results)
    count = sum(r["expected_words"] for r in results)
    if file_hash(wav_path) != audio_sha or file_hash(timing_path) != timings_sha:
        raise RuntimeError("Audio or timestamps changed during ASR audit; rerun")
    report = {"audio_sha256": audio_sha, "timestamps_sha256": timings_sha,
              "line_count": len(results), "expected_words": count, "word_errors": errors,
              "wer": round(errors / count, 5), "seconds": round(time.perf_counter() - start, 3),
              "exact_lines": sum(r["word_errors"] == 0 for r in results),
              "note": "ASR disagreements are not proof of audio defects. Not a naturalness or emotion test.",
              "rows": results}
    write_json(dest / f"{name}.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
