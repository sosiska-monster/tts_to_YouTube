import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_voice_engine import CPUVoiceEngine, write_wav

engine = CPUVoiceEngine(threads=2)
out_dir = Path("output/variant_b_options")
out_dir.mkdir(parents=True, exist_ok=True)
sr = 24000

# Conversational expressions of mockery/amusement for toxic_guy (Dave)
candidates = [
    {
        "num": 1,
        "id": "opt1_hilarious",
        "text": "Oh, this is hilarious! The show is starting! Go Karen, destroy them with facts!",
        "title": "Вариант 1: «Oh, this is hilarious!...»",
        "desc": "Живое разговорное восклицание с задором («Ой, это умора!»). Модель читает слово hilarious с естественной весёлой интонацией."
    },
    {
        "num": 2,
        "id": "opt2_pure_gold",
        "text": "Now this is pure gold! The show is starting! Go Karen, destroy them with facts!",
        "title": "Вариант 2: «Now this is pure gold!...»",
        "desc": "Классический саркастичный интернет-сленг («Вот это чистое золото!»). Идеально подходит токсичному троллю в чате."
    },
    {
        "num": 3,
        "id": "opt3_i_cant_even",
        "text": "I can't even! The show is starting! Go Karen, destroy them with facts!",
        "title": "Вариант 3: «I can't even!...»",
        "desc": "Современная разговорная ирония («Я просто не могу / умираю со смеху!»). Звучит живо, естественно и по-человечески."
    },
    {
        "num": 4,
        "id": "opt4_oh_man_here_we_go",
        "text": "Oh man, here we go! The show is starting! Go Karen, destroy them with facts!",
        "title": "Вариант 4: «Oh man, here we go!...»",
        "desc": "«О боже, началось!» — живая реакция на разгорающийся скандал."
    },
    {
        "num": 5,
        "id": "opt5_yes_finally",
        "text": "Yes! The show is finally starting! Go Karen, destroy them with facts!",
        "title": "Вариант 5: «Yes! The show is finally starting!...»",
        "desc": "Чистое злорадное предвкушение без лишних слов («Да! Шоу наконец-то начинается!»)."
    }
]

comp_parts = []
timeline = []
cur_sample = 0

for c in candidates:
    audio, meta = engine.render(c["text"], "toxic_guy", "joy", pronunciation_fixes=True, trim_edges=True, warmth=True, breath=False)
    p = out_dir / f"{c['id']}.wav"
    write_wav(p, audio)
    
    cue_text = f"Option {c['num']}."
    cue_audio, _ = engine.render(cue_text, "girl_narrator", "neutral", pronunciation_fixes=True, trim_edges=True, warmth=True, breath=False)
    
    comp_parts.append(cue_audio)
    cur_sample += len(cue_audio)
    
    p_cue = np.zeros(int(sr * 0.35), dtype=np.float32)
    comp_parts.append(p_cue)
    cur_sample += len(p_cue)
    
    start_sec = cur_sample / sr
    comp_parts.append(audio)
    cur_sample += len(audio)
    
    timeline.append({
        "option": c["num"],
        "title": c["title"],
        "text": c["text"],
        "desc": c["desc"],
        "start_time": f"{int(start_sec//60):02d}:{start_sec%60:05.2f}",
        "raw_seconds": round(start_sec, 2)
    })
    
    gap = np.zeros(int(sr * 0.9), dtype=np.float32)
    comp_parts.append(gap)
    cur_sample += len(gap)

unified = np.concatenate(comp_parts)
comp_path = out_dir / "variant_b_options_comparison.wav"
write_wav(comp_path, unified)
print(f"Unified comparison track saved: {comp_path} ({len(unified)/sr:.2f}s)")

import json
with open(out_dir / "timeline.json", "w", encoding="utf-8") as f:
    json.dump(timeline, f, indent=2, ensure_ascii=False)
