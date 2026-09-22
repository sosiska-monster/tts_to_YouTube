import sys
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_voice_engine import CPUVoiceEngine, write_wav, read_wav, apply_analog_warmth, trim_outer_silence, finish_audio

engine = CPUVoiceEngine(threads=2)
out_dir = Path("output/real_laugh_options")
out_dir.mkdir(parents=True, exist_ok=True)

sr = 24000

# 1. Synthesize the clean phrase without any text laugh:
# "The show is starting! Go Karen, destroy them with facts!"
speech_text = "The show is starting! Go Karen, destroy them with facts!"
speech_audio, meta = engine.render(speech_text, "toxic_guy", "joy", pronunciation_fixes=True, trim_edges=True, warmth=True, breath=False)

def match_timbre(audio_sample, target_ref):
    # Match RMS level to am_fenrir
    rms_target = np.sqrt(np.mean(target_ref**2))
    rms_sample = np.sqrt(np.mean(audio_sample**2))
    if rms_sample > 0:
        audio_sample = audio_sample * (rms_target * 0.95 / rms_sample)
    
    # Simple high-frequency presence boost (discrete high-pass blend) to match Kokoro crispness
    diff = np.diff(audio_sample, prepend=audio_sample[0])
    matched = audio_sample + 0.35 * diff
    
    # Project's standard analog harmonic saturation
    matched = apply_analog_warmth(matched, drive=0.06)
    return matched.astype(np.float32)

# Load our 24kHz source laughs using read_wav
y_jody = read_wav("output/chuckle_jody_24k.wav")
laugh_a = y_jody[int(sr * 0.15):int(sr * 1.10)]
laugh_a = match_timbre(laugh_a, speech_audio)

y_snow = read_wav("output/chuckles_snow_24k.wav")
laugh_b = y_snow[int(sr * 2.10):int(sr * 2.85)]
laugh_b = match_timbre(laugh_b, speech_audio)

laugh_c = y_snow[int(sr * 0.10):int(sr * 1.10)]
laugh_c = match_timbre(laugh_c, speech_audio)

laugh_d = y_snow[int(sr * 4.30):int(sr * 5.05)]
laugh_d = match_timbre(laugh_d, speech_audio)

def splice_laugh(laugh_clip, speech_clip, pause_ms=120):
    pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
    fade_len = int(sr * 0.02)
    l = laugh_clip.copy()
    if len(l) > fade_len:
        l[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    s = speech_clip.copy()
    if len(s) > fade_len:
        s[:fade_len] *= np.linspace(0, 1, fade_len)
        
    combined = np.concatenate([l, pause, s])
    return finish_audio(combined)

sample_options = [
    {
        "num": 1,
        "id": "real_sample_1_relaxed_mild",
        "title": "Вариант 1 (Сэмпл): Непринуждённый мягкий мужской смешок",
        "desc": "Живой сэмпл реального мужского смешка (негромкий, спокойный, абсолютно естественный выдох-смех), согласованный по тембру и частотам с голосом Dave.",
        "audio": splice_laugh(laugh_a, speech_audio, pause_ms=100)
    },
    {
        "num": 2,
        "id": "real_sample_2_snicker",
        "title": "Вариант 2 (Сэмпл): Короткий ехидный смешок (snicker)",
        "desc": "Живой мужской ироничный смешок в нос / ухмылка на 0.75 с. Звучит цинично и живо для токсичного персонажа.",
        "audio": splice_laugh(laugh_b, speech_audio, pause_ms=90)
    },
    {
        "num": 3,
        "id": "real_sample_3_energetic",
        "title": "Вариант 3 (Сэмпл): Раскатистый живой смешок из 3-4 тактов",
        "desc": "Живой открытый человеческий смех с естественными импульсами диафрагмы.",
        "audio": splice_laugh(laugh_c, speech_audio, pause_ms=110)
    },
    {
        "num": 4,
        "id": "real_sample_4_subtle_throat",
        "title": "Вариант 4 (Сэмпл): Приглушённый гортанный смех с затуханием",
        "desc": "Глуховатый, спокойный человеческий смех с естественным спадом высоты тона.",
        "audio": splice_laugh(laugh_d, speech_audio, pause_ms=100)
    }
]

# Save individual wavs
for opt in sample_options:
    p = out_dir / f"{opt['id']}.wav"
    write_wav(p, opt["audio"])
    print(f"[{opt['id']}] Saved {len(opt['audio'])/sr:.2f}s")

# Build comparison track with announcer cues
comp_parts = []
timeline = []
cur_sample = 0

for opt in sample_options:
    cue_text = f"Option {opt['num']}."
    cue_audio, _ = engine.render(cue_text, "girl_narrator", "neutral", pronunciation_fixes=True, trim_edges=True, warmth=True, breath=False)
    
    comp_parts.append(cue_audio)
    cur_sample += len(cue_audio)
    
    p_cue = np.zeros(int(sr * 0.4), dtype=np.float32)
    comp_parts.append(p_cue)
    cur_sample += len(p_cue)
    
    start_sec = cur_sample / sr
    comp_parts.append(opt["audio"])
    cur_sample += len(opt["audio"])
    
    timeline.append({
        "option": opt["num"],
        "title": opt["title"],
        "start_time": f"{int(start_sec//60):02d}:{start_sec%60:05.2f}",
        "raw_seconds": round(start_sec, 2)
    })
    
    gap = np.zeros(int(sr * 1.0), dtype=np.float32)
    comp_parts.append(gap)
    cur_sample += len(gap)

unified = np.concatenate(comp_parts)
comp_path = out_dir / "real_laughter_samples_comparison.wav"
write_wav(comp_path, unified)
print(f"\nUnified comparison track saved: {comp_path} ({len(unified)/sr:.2f}s)")

import json
with open(out_dir / "samples_timeline.json", "w", encoding="utf-8") as f:
    json.dump(timeline, f, indent=2, ensure_ascii=False)
