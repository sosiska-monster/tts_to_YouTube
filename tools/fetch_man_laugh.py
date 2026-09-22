import urllib.request
import librosa
import soundfile as sf
import numpy as np

url = "https://ia601600.us.archive.org/27/items/Red_Library_Voices_Men_1/R15-32-Man%20Laughing%20Heartily.mp3"
headers = {"User-Agent": "Mozilla/5.0"}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp, open("output/man_laugh.mp3", "wb") as f:
    f.write(resp.read())

y, sr = librosa.load("output/man_laugh.mp3", sr=24000, mono=True)
print(f"Downloaded man laugh: length={len(y)/sr:.2f}s, sr={sr}")
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C5"), sr=sr)
valid_f0 = f0[~np.isnan(f0)]
print("Valid F0 frames:", len(valid_f0))
if len(valid_f0):
    print(f"Median F0: {np.median(valid_f0):.1f} Hz, Mean: {np.mean(valid_f0):.1f} Hz, Range: {np.min(valid_f0):.1f} - {np.max(valid_f0):.1f} Hz")
