"""Download the explicit CPU runtime assets without changing the legacy engine."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "cpu_audio"
ASSETS = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in ASSETS.items():
        target = DEST / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".download")
            print(f"Downloading {name}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "cpu-audio-setup/1"})
            with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            partial.replace(target)
        minimum = 100_000_000 if name.endswith("onnx") else 1_000_000
        if target.stat().st_size < minimum:
            raise RuntimeError(f"Unexpected asset size: {target}")
        records[name] = {"source": url, "bytes": target.stat().st_size,
                         "sha256": sha256(target)}
        print(name, records[name], flush=True)
    (DEST / "downloads.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
