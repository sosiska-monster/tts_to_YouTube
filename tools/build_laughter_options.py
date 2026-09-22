import sys
import wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpu_voice_engine import CPUVoiceEngine, write_wav

engine = CPUVoiceEngine(threads=2)
out_dir = Path("output/laughter_variants")
out_dir.mkdir(parents=True, exist_ok=True)

# We explore diverse rhythms, vowels, speeds, and phrasing styles for toxic_guy (Dave)
# Core phrase: "...the show is starting! Go Karen, destroy them with facts!"
tail_phrase = "the show is starting! Go Karen, destroy them with facts!"
tok = getattr(engine.tts, "tokenizer", engine.tts)
tail_ph = tok.phonemize(tail_phrase, "en-us")

variants = [
    {
        "id": "variant_1_smirk_heh",
        "title": "Вариант 1: Саркастичный смешок-ухмылка (Heh...)",
        "desc": "Короткий, уверенный язвительный полусмешок с мягкой паузой перед фразой. Звучит очень естественно для токсичного персонажа.",
        "phonemes": "hˈɛh... " + tail_ph,
        "speed": 0.96,
        "text": "Heh... the show is starting! Go Karen, destroy them with facts!"
    },
    {
        "id": "variant_2_calm_two_beat",
        "title": "Вариант 2: Размеренный двойной смех (Ha-ha...)",
        "desc": "Не торопливый смех с паузой между смешками (темп 0.92, без пулемётной спешки).",
        "phonemes": "hˈʌ... hʌ, " + tail_ph,
        "speed": 0.92,
        "text": "Haha... the show is starting! Go Karen, destroy them with facts!"
    },
    {
        "id": "variant_3_open_three_beat",
        "title": "Вариант 3: Классический тройной смех с расстановкой (Ha, ha, ha!)",
        "desc": "3 раздельных смешка, но в мягком темпе через запятые, а не роботизированный крик.",
        "phonemes": "hˈɑː, hɑː, hɑː, " + tail_ph,
        "speed": 0.94,
        "text": "Ha, ha, ha! The show is starting! Go Karen, destroy them with facts!"
    },
    {
        "id": "variant_4_exclamatory_oh_man",
        "title": "Вариант 4: Живая разговорная реакция (Oh man, haha!)",
        "desc": "В живой речи вместо сухого 'хаха' люди часто реагируют восклицанием с посмеиванием. Звучит максимально по-человечески.",
        "phonemes": "oʊ mˈæn, hɑːhˈɑː! " + tail_ph,
        "speed": 0.98,
        "text": "Oh man, haha! The show is starting! Go Karen, destroy them with facts!"
    },
    {
        "id": "variant_5_pfft_snicker",
        "title": "Вариант 5: Смешок сквозь зубы / фырканье (Pfft, haha!)",
        "desc": "Ироничный фыркающий смешок сдавленного смеха, очень подходит под злорадство.",
        "phonemes": "pfˈt, hɑːhˈɑː, " + tail_ph,
        "speed": 0.95,
        "text": "Pfft, haha! The show is starting! Go Karen, destroy them with facts!"
    },
    {
        "id": "variant_6_relaxed_chuckle",
        "title": "Вариант 6: Мягкое гортанное покашливание-смех (He-he...)",
        "desc": "Глуховатый, спокойный смешок с понижением интонации.",
        "phonemes": "hˈɛ... hɛ... " + tail_ph,
        "speed": 0.90,
        "text": "Hehe... the show is starting! Go Karen, destroy them with facts!"
    }
]

catalog = []
rendered_audios = []

for idx, var in enumerate(variants):
    audio, sr = engine.tts.create(var["phonemes"], voice="toxic_guy", speed=var["speed"], is_phonemes=True, trim=False)
    # subtle warmth
    from cpu_voice_engine import apply_analog_warmth, trim_outer_silence, finish_audio
    audio, _ = trim_outer_silence(audio)
    audio = apply_analog_warmth(audio)
    audio = finish_audio(audio)
    
    wav_path = out_dir / f"{var['id']}.wav"
    write_wav(wav_path, audio)
    
    catalog.append({
        "num": idx + 1,
        "id": var["id"],
        "title": var["title"],
        "desc": var["desc"],
        "text": var["text"],
        "duration": round(len(audio) / sr, 2),
        "file": str(wav_path)
    })
    rendered_audios.append((idx + 1, var["title"], audio))
    print(f"[{var['id']}] Generated {len(audio)/sr:.2f}s")

# Now let's assemble a unified comparison audio track with announcer cues: "Option 1", "Option 2", etc.
sr = 24000
comp_parts = []
time_marks = []
cur_sample = 0

for num, title, audio in rendered_audios:
    # Announce option number using girl_narrator
    cue_text = f"Option {num}."
    cue_audio, _ = engine.render(cue_text, "girl_narrator", "neutral", pronunciation_fixes=True, trim_edges=True, warmth=True, breath=False)
    
    start_sec = cur_sample / sr
    comp_parts.append(cue_audio)
    cur_sample += len(cue_audio)
    
    # 0.4s pause
    pause_cue = np.zeros(int(sr * 0.4), dtype=np.float32)
    comp_parts.append(pause_cue)
    cur_sample += len(pause_cue)
    
    # Toxic guy's utterance
    opt_start_sec = cur_sample / sr
    comp_parts.append(audio)
    cur_sample += len(audio)
    opt_end_sec = cur_sample / sr
    
    time_marks.append({
        "option": num,
        "title": title,
        "start_time": f"{int(opt_start_sec//60):02d}:{opt_start_sec%60:05.2f}",
        "raw_seconds": round(opt_start_sec, 2)
    })
    
    # 1.0s gap between options
    gap = np.zeros(int(sr * 1.0), dtype=np.float32)
    comp_parts.append(gap)
    cur_sample += len(gap)

unified_audio = np.concatenate(comp_parts)
comp_path = out_dir / "all_laughter_options_comparison.wav"
write_wav(comp_path, unified_audio)
print(f"\nUnified comparison track saved: {comp_path} ({len(unified_audio)/sr:.2f}s)")

import json
report_path = out_dir / "laughter_options.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({"variants": catalog, "timeline": time_marks}, f, indent=2, ensure_ascii=False)
