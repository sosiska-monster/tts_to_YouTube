"""Build the portable audio runtime and listening deliverables; no legacy writes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpu_voice_engine import (read_json, write_json, read_wav, write_wav, file_hash,
                              normalize_text, SAMPLE_RATE, PREVIEW_LINES)

OUT = ROOT / "output" / "cpu_audio"


def validate_delivery(master, timestamps, benchmark, audit, story):
    """Fail closed: never attach an unbound historical ASR report to a release."""
    rows = read_json(timestamps)
    if benchmark.get("master_sha256") != file_hash(master):
        raise ValueError("Benchmark does not match master audio")
    if audit.get("audio_sha256") != file_hash(master):
        raise ValueError("ASR audio checksum missing or stale; rerun tools/qa/audit_cpu_audio.py")
    if audit.get("timestamps_sha256") != file_hash(timestamps):
        raise ValueError("ASR timestamps checksum missing or stale; rerun audit")
    if len(rows) != len(story) or [r["line_index"] for r in rows] != list(range(len(story))):
        raise ValueError("Missing, duplicated, or reordered story lines")
    if audit.get("line_count") != len(rows) or len(audit.get("rows", [])) != len(rows):
        raise ValueError("Incomplete ASR audit")
    audio = read_wav(master)
    previous_end = 0
    for row, expected, checked in zip(rows, story, audit["rows"]):
        if any(row[k] != expected[k] for k in ("sender_id", "emotion", "text")):
            raise ValueError("Master does not match current story")
        if (checked["line_index"] != row["line_index"] or
                checked["expected"] != normalize_text(row["text"]) or
                checked["character"] != row["sender_id"] or checked["emotion"] != row["emotion"]):
            raise ValueError("ASR row does not match timestamps")
        start, end = [round(row[k] * SAMPLE_RATE / 1000) for k in ("start_ms", "end_ms")]
        if not (previous_end <= start < end <= len(audio)):
            raise ValueError("Invalid or overlapping audio intervals")
        previous_end = end
    if np.any(np.abs(audio) >= 0.999):
        raise ValueError("Master contains clipped samples")
    return audio, rows


def installed_license(distribution):
    package = importlib.metadata.distribution(distribution)
    for entry in package.files or []:
        if Path(str(entry)).name.upper() in ("LICENSE", "LICENSE.TXT", "LICENSE.MD"):
            return Path(package.locate_file(entry))
    raise FileNotFoundError(f"No installed license found for {distribution}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery-dir", type=Path, required=True)
    args = parser.parse_args()
    before = read_wav(ROOT / "output" / "final_voiceover.wav")
    before_times = read_json(ROOT / "output" / "dialog_timestamps.json")
    audit = read_json(OUT / "cpu_asr_audit.json")
    after, after_rows = validate_delivery(
        OUT / "final_voiceover_cpu.wav", OUT / "final_voiceover_cpu_timestamps.json",
        read_json(OUT / "final_voiceover_cpu_benchmark.json"), audit, read_json(ROOT / "story.json"))
    after_times = {r["line_index"]: r for r in after_rows}
    parts, timeline, offset = [], [], 0
    for i in PREVIEW_LINES:
        for label, signal, row in [("before", before, before_times[i]), ("after", after, after_times[i])]:
            start, end = [round(row[k] * SAMPLE_RATE / 1000) for k in ["start_ms", "end_ms"]]
            if not 0 <= start < end <= len(signal):
                raise ValueError(f"Invalid comparison interval: {i}/{label}")
            if row["text"] != after_times[i]["text"]:
                raise ValueError(f"Comparison text differs: line {i}")
            segment = signal[start:end]
            timeline.append({"line_index": i, "variant": label, "character": row["sender_id"],
                             "emotion": row["emotion"], "text": row["text"],
                             "start_ms": round(offset * 1000 / SAMPLE_RATE),
                             "end_ms": round((offset + len(segment)) * 1000 / SAMPLE_RATE)})
            gap = np.zeros(round(SAMPLE_RATE * (0.6 if label == "before" else 1.0)), np.float32)
            parts.extend([segment, gap])
            offset += len(segment) + len(gap)
    write_wav(OUT / "comparison_before_after.wav", np.concatenate(parts))
    write_json(OUT / "comparison_before_after.json", timeline)
    licenses = OUT / "licenses"
    licenses.mkdir(exist_ok=True)
    target_license = licenses / "Apache-2.0.txt"
    if not target_license.exists():
        req = urllib.request.Request("https://www.apache.org/licenses/LICENSE-2.0", headers={"User-Agent": "cpu-audio/1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode("utf-8")
        if "TERMS AND CONDITIONS" not in text:
            raise ValueError("Expected plain Apache license")
        target_license.write_text(text, encoding="utf-8")
    shutil.copy2(installed_license("kokoro-onnx"), licenses / "kokoro-onnx-MIT.txt")
    notices = ("Third-party assets\n"
               "Kokoro-82M weights and voice tensors: hexgrad, Apache-2.0.\n"
               "https://huggingface.co/hexgrad/Kokoro-82M\n"
               "ONNX export: thewh1teagle/kokoro-onnx.\n"
               "https://github.com/thewh1teagle/kokoro-onnx\n"
               "Voice matrices were selected and repackaged as six named characters; values unchanged.\n"
               "Model bytes unchanged from source release. Checksums in downloads.json.\n"
               "The model card attributes Koniwa tnc (CC BY 3.0) and SIWIS (CC BY 4.0) training material:\n"
               "https://github.com/koniwa/koniwa\n"
               "https://datashare.ed.ac.uk/handle/10283/2353\n"
               "Python packages are not vendored. Their licenses accompany installation, including espeak-ng.\n")
    (licenses / "NOTICE.txt").write_text(notices, encoding="utf-8")
    paths = ["cpu_voice_engine.py", "requirements-cpu.txt", "story.json", "configs/cpu_voices.json",
             "models/cpu_audio/kokoro-v1.0.onnx", "models/cpu_audio/downloads.json",
             "voice_profiles/cpu/characters.npz", "voice_profiles/cpu/manifest.json"]
    archive = OUT / "cpu_host_bundle.zip"
    payload = {relative: ROOT / relative for relative in paths}
    payload["INSTRUCTIONS_RU.txt"] = OUT / "ИНСТРУКЦИЯ_CPU.txt"
    payload.update({"licenses/" + path.name: path for path in sorted(licenses.iterdir()) if path.is_file()})
    members = {name: file_hash(path) for name, path in payload.items()}
    temporary_archive = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name, path in payload.items():
            z.write(path, name)
        z.writestr("bundle_checksums.json", json.dumps(members, indent=2))
    with zipfile.ZipFile(temporary_archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC validation failed")
        for name, expected_hash in members.items():
            digest = hashlib.sha256()
            with z.open(name) as entry:
                for block in iter(lambda: entry.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"Archive checksum mismatch: {name}")
    temporary_archive.replace(archive)
    summary = {"audio_duration_seconds": len(after) / SAMPLE_RATE, "audio_peak": float(np.max(np.abs(after))),
               "archive_bytes": archive.stat().st_size, "profile_bytes": (ROOT / "voice_profiles/cpu/characters.npz").stat().st_size,
               "master_sha256": file_hash(OUT / "final_voiceover_cpu.wav"), "archive_sha256": file_hash(archive),
               "asr_wer": audit["wer"], "asr_exact_lines": audit["exact_lines"], "line_count": len(after_rows),
               "status": "cpu_deployment_candidate_not_human_indistinguishability_pass"}
    write_json(OUT / "delivery_checks.json", summary)
    args.delivery_dir.mkdir(parents=True, exist_ok=True)
    deliver = ["preview.wav", "final_voiceover_cpu.wav", "comparison_before_after.wav", "comparison_before_after.json",
               "cpu_host_bundle.zip", "ИНСТРУКЦИЯ_CPU.txt", "final_voiceover_cpu_benchmark.json", "cpu_asr_audit.json",
               "final_voiceover_cpu_timestamps.json", "delivery_checks.json"]
    for name in deliver:
        if (OUT / name).resolve() != (args.delivery_dir / name).resolve():
            shutil.copy2(OUT / name, args.delivery_dir / name)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
