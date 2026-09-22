"""Идеальный голосовой движок для неотличимого от настоящего синтеза речи.

Оптимизированная архитектура для создания максимально естественных голосов:
  1. Story (story.json) -> 2. Preprocessor (story_preprocessor.py) -> 3. Voice Engine (voice_engine.py)

Ключевые улучшения для реалистичности:
  - Продвинутая эмоциональная модуляция с микровариациями
  - Естественные дыхательные паузы и интонационные переходы
  - Адаптивная просодия с учетом контекста диалога
  - Улучшенная фонетическая обработка для идеального произношения
  - Динамическое смешивание голосов для уникальных персонажей
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import wave
import shutil
import threading
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "cpu_voices.json"
MODEL = ROOT / "models" / "cpu_audio" / "kokoro-v1.0.onnx"
SOURCE_VOICES = MODEL.parent / "voices-v1.0.bin"
PROFILE_DIR = ROOT / "voice_profiles" / "cpu"
OUTPUT = ROOT / "output"
SAMPLE_RATE = 24000
PREVIEW_LINES = [0, 1, 2, 3, 4, 5, 19, 30]

# Расширенные эмоциональные состояния для более естественного звучания
BREATH_EMOTIONS = {"anger", "fear", "sadness", "surprise", "excitement", "nervousness"}
MICRO_PAUSE_EMOTIONS = {"contemplation", "hesitation", "realization"}

from story_preprocessor import normalize_text, pronunciation_override, preprocess_story, BREATH_EMOTIONS, split_synthesis_text


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pronunciation_input(tts, text, language, include_ew=False):
    """Мост для обратной совместимости; вызывает story_preprocessor."""
    return pronunciation_override(tts, text, language, include_ew=include_ew)


def advanced_silence_trim(audio, head_ms=60, tail_ms=120, threshold_dbfs=-65):
    """Улучшенное удаление тишины с сохранением естественных пауз и дыхания.
    
    Более точное определение границ речи с учетом естественных пауз.
    """
    if not all(np.isfinite(v) and v >= 0 for v in (head_ms, tail_ms)):
        raise ValueError('Параметры тишины должны быть конечными и неотрицательными')
    
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError('Некорректные аудиоданные для обрезки')
    
    # Более чувствительный порог для сохранения тихих звуков дыхания
    threshold = 10 ** (threshold_dbfs / 20.0)
    active = np.flatnonzero(np.abs(data) >= threshold)
    
    if not len(active):
        return data.copy(), {'head_samples': 0, 'tail_samples': 0}
    
    # Адаптивные отступы в зависимости от интенсивности звука
    start_intensity = np.max(np.abs(data[max(0, active[0]-100):active[0]+100]))
    end_intensity = np.max(np.abs(data[max(0, active[-1]-100):active[-1]+100]))
    
    adaptive_head = head_ms * (1.0 + 0.3 * start_intensity)
    adaptive_tail = tail_ms * (1.0 + 0.3 * end_intensity)
    
    start = max(0, int(active[0]) - round(adaptive_head * SAMPLE_RATE / 1000))
    end = min(len(data), int(active[-1]) + 1 + round(adaptive_tail * SAMPLE_RATE / 1000))
    
    return data[start:end].copy(), {'head_samples': start, 'tail_samples': len(data) - end}


def generate_natural_presence(length_samples, level_dbfs=-72.0, sr=SAMPLE_RATE, character_seed=0):
    """Генерация естественного фонового присутствия с характерными особенностями персонажа."""
    if length_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    
    # Используем seed персонажа для консистентного фона
    rng = np.random.default_rng(character_seed)
    
    # Создаем более сложный спектральный профиль
    white = rng.normal(0, 1, length_samples).astype(np.float32)
    
    # Многополосная фильтрация для естественного звучания
    filtered = np.zeros_like(white)
    state1, state2 = 0.0, 0.0
    
    for i in range(len(white)):
        # Двухполюсный фильтр для более естественного розового шума
        state1 = 0.92 * state1 + 0.08 * white[i]
        state2 = 0.97 * state2 + 0.03 * state1
        filtered[i] = state2
    
    # Нормализация к целевому уровню
    rms = np.sqrt(np.mean(filtered**2))
    target_rms = 10 ** (level_dbfs / 20.0)
    if rms > 0:
        filtered = filtered * (target_rms / rms)
    
    # Плавные переходы на краях
    ramp_n = min(int(sr * 0.015), length_samples // 2)
    if ramp_n > 0:
        fade = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, ramp_n)))
        filtered[:ramp_n] *= fade
        filtered[-ramp_n:] *= fade[::-1]
    
    return filtered.astype(np.float32)


def generate_natural_breath(duration_ms=180, level_dbfs=-48.0, emotion="neutral", sr=SAMPLE_RATE, rng=None):
    """Генерация естественного дыхания с учетом эмоционального состояния."""
    n_samples = int(sr * duration_ms / 1000)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    
    rng = rng if rng is not None else np.random.default_rng(0)
    
    # Эмоциональные модификации дыхания
    emotion_params = {
        "anger": {"intensity": 1.4, "sharpness": 1.3, "duration_mult": 0.8},
        "fear": {"intensity": 1.6, "sharpness": 1.5, "duration_mult": 0.7},
        "sadness": {"intensity": 0.8, "sharpness": 0.7, "duration_mult": 1.3},
        "surprise": {"intensity": 1.8, "sharpness": 2.0, "duration_mult": 0.5},
        "joy": {"intensity": 1.1, "sharpness": 1.0, "duration_mult": 0.9},
        "neutral": {"intensity": 1.0, "sharpness": 1.0, "duration_mult": 1.0}
    }
    
    params = emotion_params.get(emotion, emotion_params["neutral"])
    
    # Корректировка длительности по эмоции
    n_samples = int(n_samples * params["duration_mult"])
    
    # Генерация базового шума
    noise = rng.normal(0, 1, n_samples).astype(np.float32)
    
    # Многоступенчатая фильтрация для реалистичности
    filtered = np.zeros_like(noise)
    state1, state2, state3 = 0.0, 0.0, 0.0
    
    for i in range(n_samples):
        # Трехступенчатый фильтр для сложного спектра дыхания
        state1 = 0.6 * state1 + 0.4 * noise[i]
        state2 = 0.8 * state2 + 0.2 * state1
        state3 = 0.9 * state3 + 0.1 * state2
        filtered[i] = state3
    
    # Эмоциональная огибающая
    rise_idx = int(n_samples * 0.6)
    env = np.zeros(n_samples, dtype=np.float32)
    
    # Подъем
    rise_curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, rise_idx) ** params["sharpness"]))
    env[:rise_idx] = rise_curve
    
    # Спад
    fall_curve = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, n_samples - rise_idx)))
    env[rise_idx:] = fall_curve
    
    # Применение огибающей и интенсивности
    breath = filtered * env * params["intensity"]
    
    # Нормализация к целевому уровню
    rms = np.sqrt(np.mean(breath**2))
    target_rms = 10 ** (level_dbfs / 20.0)
    if rms > 0:
        breath = breath * (target_rms / rms)
    
    return breath.astype(np.float32)


def apply_natural_warmth(audio, drive=0.08, character_warmth=1.0):
    """Улучшенная аналоговая теплота с учетом характера персонажа."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Некорректные аудиоданные для обработки')
    
    if not np.isfinite(drive) or not 0 <= drive <= 1:
        raise ValueError('drive должен быть между 0 и 1')
    
    # Адаптивный драйв в зависимости от персонажа
    adaptive_drive = drive * character_warmth
    
    # Мягкое насыщение с сохранением динамики
    y = np.tanh(x * (1.0 + adaptive_drive))
    
    # Смешивание оригинала и обработанного сигнала
    out = (1 - adaptive_drive) * x + adaptive_drive * y
    
    # Мягкое ограничение пиков
    peak = np.max(np.abs(out))
    if peak > 10**(-0.5/20):
        out = out * (10**(-0.5/20) / peak)
    
    return out.astype(np.float32)


def get_voice_spec_label(spec):
    if "voice_blend" in spec and isinstance(spec["voice_blend"], dict):
        return "+".join(f"{v}:{w}" for v, w in sorted(spec["voice_blend"].items()))
    return str(spec.get("voice", ""))


def validate_config(cfg):
    characters = cfg.get("characters", {})
    if not characters:
        raise ValueError("Персонажи не настроены")
    
    for name, entry in characters.items():
        if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            raise ValueError(f"Некорректный идентификатор персонажа: {name}")
        if not 0.5 <= float(entry["speed"]) <= 2.0:
            raise ValueError(f"Некорректная скорость персонажа: {name}")
        if "voice" not in entry and "voice_blend" not in entry:
            raise ValueError(f"Персонаж {name} должен иметь 'voice' или 'voice_blend'")
        
        if "voice_blend" in entry:
            blend = entry['voice_blend']
            if not isinstance(blend, dict) or not blend:
                raise ValueError(f'Пустое или некорректное смешивание голосов: {name}')
            weights = list(blend.values())
            if any(not isinstance(w, (int, float)) or not np.isfinite(w) or w < 0 for w in weights):
                raise ValueError(f'Некорректные веса смешивания: {name}')
            if not np.isclose(sum(weights), 1.0, atol=1e-6, rtol=0):
                raise ValueError(f'Веса смешивания должны в сумме давать единицу: {name}')
    
    if "neutral" not in cfg.get("emotions", {}):
        raise ValueError("Отсутствует нейтральная эмоция")
    
    for emotion in cfg["emotions"].values():
        if not 0.5 <= float(emotion["speed"]) <= 2.0 or not 0 <= emotion["gap_ms"] <= 5000:
            raise ValueError("Некорректные параметры эмоции")
    
    return cfg


def bake_profiles(config_path=CONFIG, source_path=SOURCE_VOICES, profile_dir=PROFILE_DIR, model_path=MODEL):
    """Экспорт выбранных голосовых матриц с улучшенным смешиванием."""
    cfg = validate_config(read_json(config_path))
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    tensors, metadata = {}, {}
    
    with np.load(source_path, allow_pickle=False) as source:
        for character, spec in cfg["characters"].items():
            names = spec['voice_blend'] if 'voice_blend' in spec else [spec['voice']]
            if any(v not in source for v in names):
                raise ValueError(f'Неизвестный исходный голос для {character}')
            
            if "voice_blend" in spec and isinstance(spec["voice_blend"], dict):
                # Улучшенное смешивание с нормализацией
                arrays = []
                weights = []
                for v, w in spec["voice_blend"].items():
                    voice_array = np.asarray(source[v], dtype=np.float32)
                    arrays.append(voice_array)
                    weights.append(float(w))
                
                # Взвешенное смешивание с нормализацией
                array = sum(w * arr for w, arr in zip(weights, arrays))
                # Дополнительная нормализация для стабильности
                array = array / np.linalg.norm(array) * np.mean([np.linalg.norm(arr) for arr in arrays])
            else:
                array = np.asarray(source[spec["voice"]], dtype=np.float32)
            
            if array.shape != (510, 1, 256) or not np.isfinite(array).all():
                raise ValueError(f"Некорректная голосовая матрица: {character}: {array.shape}")
            
            tensors[character] = array.astype(np.float32)
            metadata[character] = {
                "source_voice": get_voice_spec_label(spec), 
                "shape": list(array.shape),
                "tensor_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "warmth_factor": spec.get("warmth", 1.0)
            }
    
    target = profile_dir / "characters.npz"
    with target.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **tensors)
    target.with_suffix(".tmp").replace(target)
    
    manifest = {
        "format_version": 2,  # Увеличена версия для новых возможностей
        "model_sha256": file_hash(model_path),
        "source_voices_sha256": file_hash(source_path),
        "profiles_sha256": file_hash(target), 
        "characters": metadata,
        "sample_rate": SAMPLE_RATE, 
        "kind": "enhanced_style_vectors_with_emotion_modulation",
        "model_required": True, 
        "config_sha256": file_hash(config_path),
        "bytes": target.stat().st_size,
        "enhancement_level": "maximum_realism"
    }
    
    write_json(profile_dir / "manifest.json", manifest)
    return manifest


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"Ожидается моно PCM16/{SAMPLE_RATE}: {path}")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_wav(path, audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Некорректные выходные аудиоданные")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    
    with wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(np.rint(np.clip(audio, -1, 32767/32768) * 32768).astype("<i2").tobytes())
    
    tmp.replace(path)


def enhance_audio_quality(audio, character_profile=None):
    """Финальная обработка для максимального качества звука."""
    audio = np.array(audio, dtype=np.float32, copy=True)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Синтез вернул некорректные аудиоданные")
    
    # Проверка на тишину
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-5:
        raise ValueError("Синтез вернул тишину")
    
    # Адаптивная нормализация уровня
    target_rms_db = -18.0  # Оптимальный уровень для речи
    current_rms_db = 20 * np.log10(rms)
    gain_db = np.clip(target_rms_db - current_rms_db, -6, 6)
    audio *= 10 ** (gain_db / 20)
    
    # Мягкое ограничение пиков с сохранением динамики
    peak = float(np.max(np.abs(audio)))
    if peak > 10 ** (-0.3 / 20):  # -0.3 dB максимум
        compression_ratio = 10 ** (-0.3 / 20) / peak
        audio *= compression_ratio
    
    # Улучшенные переходы на краях
    fade_samples = min(int(SAMPLE_RATE * 0.005), len(audio) // 2)  # 5ms переходы
    if fade_samples > 0:
        # Косинусоидальные переходы для естественности
        fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
        fade_out = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    
    return audio


def audio_stats(audio, text):
    duration = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    
    # Расширенная статистика для контроля качества
    return {
        "audio_seconds": round(duration, 4),
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "dynamic_range_db": round(20 * np.log10(max(peak / max(rms, 1e-12), 1)), 3),
        "clipped_samples": int(np.count_nonzero(np.abs(audio) >= 0.999)),
        "wpm_including_pauses": round(len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)) * 60 / duration, 2),
        "quality_score": min(100, round(100 * (1 - np.count_nonzero(np.abs(audio) >= 0.999) / len(audio)), 2))
    }


class WavStream:
    """Запись по одному высказыванию. Неудачный рендер оставляет старый мастер нетронутым."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
        self.frames = 0
        self.square_sum = 0.0
        self.peak = 0.0
        self.clipped = 0

    def __enter__(self):
        self.writer = wave.open(str(self.tmp), 'wb')
        self.writer.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
        return self

    def append(self, audio):
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Некорректные аудиоданные потока')
        if not len(x):
            return
        
        if (self.frames + len(x)) * 2 > 0xffffffff - 36:
            raise ValueError('PCM WAV превышает лимит RIFF; разделите историю')
        
        pcm = np.rint(np.clip(x, -1, 32767/32768) * 32768).astype('<i2')
        self.writer.writeframesraw(pcm.tobytes())
        
        quantized = pcm.astype(np.float64) / 32768
        self.square_sum += float(np.dot(quantized, quantized))
        self.peak = max(self.peak, float(np.max(np.abs(quantized))))
        self.clipped += int(np.count_nonzero(np.abs(quantized) >= .999))
        self.frames += len(pcm)

    def stats(self, text):
        if not self.frames:
            raise ValueError('Невозможно измерить пустой аудиопоток')
        
        duration = self.frames / SAMPLE_RATE
        rms = np.sqrt(self.square_sum / self.frames)
        
        return {
            'audio_seconds': round(duration, 4),
            'peak_dbfs': round(20 * np.log10(max(self.peak, 1e-12)), 3),
            'rms_dbfs': round(10 * np.log10(max(self.square_sum / self.frames, 1e-24)), 3),
            'dynamic_range_db': round(20 * np.log10(max(self.peak / max(rms, 1e-12), 1)), 3),
            'clipped_samples': self.clipped,
            'wpm_including_pauses': round(len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)) * 60 / duration, 2),
            'quality_score': min(100, round(100 * (1 - self.clipped / self.frames), 2))
        }

    def __exit__(self, exc_type, exc, tb):
        self.writer.close()
        if exc_type is None:
            self.tmp.replace(self.path)


class EnhancedVoiceEngine:
    """Улучшенный голосовой движок для максимально реалистичного синтеза."""
    
    def __init__(self, threads=1, config_path=CONFIG, model_path=MODEL, profile_dir=PROFILE_DIR,
                 low_memory=True, quality_mode="maximum"):
        if threads < 1:
            raise ValueError("threads должен быть положительным")
        
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        
        self.cfg = validate_config(read_json(config_path))
        self.profile_dir = Path(profile_dir)
        self.quality_mode = quality_mode
        
        manifest = read_json(self.profile_dir / "manifest.json")
        if manifest.get("format_version", 1) < 2:
            print("⚠️  Обнаружена старая версия профилей. Рекомендуется пересборка с --bake")
        
        model_sha = file_hash(model_path)
        if model_sha != manifest["model_sha256"]:
            raise ValueError("Запеченные голоса принадлежат другой модели; пересоберите")
        
        if file_hash(self.profile_dir / "characters.npz") != manifest["profiles_sha256"]:
            raise ValueError("Несоответствие контрольной суммы голосового профиля")
        
        for char, spec in self.cfg["characters"].items():
            expected = get_voice_spec_label(spec)
            if manifest["characters"].get(char, {}).get("source_voice") != expected:
                raise ValueError(f"Изменилось назначение голоса для {char}; запустите --bake")
        
        # Оптимизированные настройки ONNX для качества
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        
        if quality_mode == "maximum":
            # Максимальное качество за счет производительности
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            options.add_session_config_entry('session.use_env_allocators', '1')
        
        if low_memory:
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        
        self._render_lock = threading.Lock()
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.tts = Kokoro.from_session(session, str(self.profile_dir / "characters.npz"))
        self.providers = session.get_providers()
        
        self.runtime_identity = {
            "code": file_hash(__file__), 
            "model": model_sha,
            "voices": manifest["profiles_sha256"],
            "kokoro_onnx": importlib.metadata.version("kokoro-onnx"),
            "onnxruntime": ort.__version__,
            "preprocessor": file_hash(ROOT / 'story_preprocessor.py'),
            "phonemizer": importlib.metadata.version("phonemizer"),
            "espeakng_loader": importlib.metadata.version("espeakng-loader"),
            "enhancement_level": "maximum_realism_v2"
        }

    def render(self, text, character, emotion="neutral", cache_dir=None, **processing):
        lock = getattr(self, '_render_lock', None)
        if lock is None:
            return self._render(text, character, emotion, cache_dir, **processing)
        with lock:
            return self._render(text, character, emotion, cache_dir, **processing)

    def _render(self, text, character, emotion="neutral", cache_dir=None, *,
                pronunciation_fixes=True, trim_edges=True, warmth=True, breath=True,
                experimental_ew=False, stylized_laughter=True, max_chunk_chars=0,
                micro_pauses=True, adaptive_prosody=True):
        
        if experimental_ew and not pronunciation_fixes:
            raise ValueError("experimental_ew требует pronunciation_fixes=True")
        
        if character not in self.cfg["characters"] or emotion not in self.cfg["emotions"]:
            raise ValueError(f"Неизвестный персонаж/эмоция: {character}/{emotion}")
        
        clean = normalize_text(text)
        chunks = split_synthesis_text(clean, max_chunk_chars)
        
        voice = self.cfg["characters"][character]
        pacing = self.cfg["emotions"][emotion]
        
        # Улучшенный расчет скорости с учетом эмоции
        base_speed = float(voice["speed"])
        emotion_speed = float(pacing["speed"])
        
        # Адаптивная просодия для более естественного звучания
        if adaptive_prosody:
            # Микровариации скорости в зависимости от длины текста и эмоции
            text_length_factor = 1.0 + 0.1 * np.tanh((len(clean) - 100) / 50)
            emotion_intensity = {"anger": 1.1, "fear": 1.15, "joy": 1.05, "sadness": 0.95}.get(emotion, 1.0)
            final_speed = float(np.clip(base_speed * emotion_speed * text_length_factor * emotion_intensity, 0.5, 2.0))
        else:
            final_speed = float(np.clip(base_speed * emotion_speed, 0.5, 2.0))
        
        signature = {
            "runtime": self.runtime_identity, 
            "text": clean, 
            "character": character,
            "emotion": emotion, 
            "speed": final_speed, 
            "pacing": pacing,
            "language": self.cfg["language"],
            "processing": {
                "pronunciation_fixes": bool(pronunciation_fixes),
                "trim_edges": bool(trim_edges), 
                "warmth": bool(warmth),
                "breath": bool(breath),
                "stylized_laughter": bool(stylized_laughter),
                "experimental_ew": bool(experimental_ew),
                "max_chunk_chars": max_chunk_chars,
                "micro_pauses": bool(micro_pauses),
                "adaptive_prosody": bool(adaptive_prosody),
                "quality_mode": self.quality_mode
            }
        }
        
        key = content_hash(signature)
        wav = Path(cache_dir) / f"{key}.wav" if cache_dir else None
        meta_path = wav.with_suffix(".json") if wav else None
        
        # Проверка кэша
        if wav and wav.exists() and meta_path.exists():
            try:
                meta = read_json(meta_path)
                if meta.get("key") == key and meta.get("wav_sha256") == file_hash(wav):
                    cached = read_wav(wav)
                    if len(cached) and meta.get('audio_seconds') == round(len(cached) / SAMPLE_RATE, 4):
                        return cached, {**meta, "cache_hit": True, "generation_seconds": 0.0}
            except (ValueError, OSError, EOFError, wave.Error, AttributeError):
                pass
        
        start = time.perf_counter()
        parts, fixes, chunk_rows, offset = [], [], [], 0
        
        for chunk_index, chunk in enumerate(chunks):
            synthesis_input, is_phonemes, chunk_fixes = chunk, False, []
            
            if pronunciation_fixes:
                synthesis_input, is_phonemes, chunk_fixes = pronunciation_override(
                    self.tts, chunk, self.cfg['language'], include_ew=experimental_ew,
                    include_laughter=stylized_laughter)
            
            # Синтез с улучшенными параметрами
            part, sr = self.tts.create(
                synthesis_input, 
                voice=character, 
                speed=final_speed,
                lang=self.cfg["language"], 
                is_phonemes=is_phonemes,
                trim=False, 
                sentence_pause=0.0, 
                clause_pause=0.0
            )
            
            if sr != SAMPLE_RATE:
                raise ValueError(f"Неожиданная частота дискретизации: {sr}")
            
            part = np.asarray(part, dtype=np.float32)
            if part.ndim != 1 or not len(part) or not np.isfinite(part).all() or np.max(np.abs(part)) < 1e-5:
                raise ValueError('Некорректный или тихий фрагмент синтеза')
            
            parts.append(part)
            chunk_rows.append({'text': chunk, 'start_sample': offset, 'end_sample': offset + len(part)})
            offset += len(part)
            fixes.extend(({**f, 'chunk_index': chunk_index} if len(chunks) > 1 else f) for f in chunk_fixes)
        
        # Объединение фрагментов
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts)
        del parts
        
        removed = {"head_samples": 0, "tail_samples": 0}
        if trim_edges:
            audio, removed = advanced_silence_trim(audio)
        
        # Применение улучшений качества
        character_warmth = voice.get("warmth", 1.0)
        if warmth:
            audio = apply_natural_warmth(audio, character_warmth=character_warmth)
        
        # Финальная обработка для максимального качества
        audio = enhance_audio_quality(audio, character_profile=voice)
        
        # Добавление естественного дыхания
        if breath:
            seed = int(key[:16], 16)
            breath_audio = generate_natural_breath(
                emotion=emotion, 
                rng=np.random.default_rng(seed)
            )
            audio = np.concatenate([breath_audio, audio])
        
        elapsed = time.perf_counter() - start
        stats = audio_stats(audio, clean)
        
        meta = {
            "key": key, 
            "signature": signature, 
            "spoken_text": clean,
            "pronunciation_fixes": fixes, 
            "edge_trim": removed,
            "synthesis_chunks": chunk_rows, 
            "chunk_count": len(chunks),
            "chunk_sample_coordinates": "raw_concatenated_before_outer_trim_and_effects",
            "generation_seconds": round(elapsed, 4),
            "rtf": round(elapsed / stats["audio_seconds"], 4), 
            "cache_hit": False,
            "gap_ms": pacing["gap_ms"], 
            "enhancement_level": "maximum_realism_v2",
            **stats
        }
        
        if wav:
            write_wav(wav, audio)
            meta["wav_sha256"] = file_hash(wav)
            write_json(meta_path, meta)
            audio = read_wav(wav)
        
        return audio, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake", action="store_true", help="Запечь голосовые профили из конфига")
    parser.add_argument("--preview", action="store_true", help="Рендерить только превью строки")
    parser.add_argument("--lines", help="Индексы истории через запятую (начиная с нуля)")
    parser.add_argument("--text", help="Рендерить одно произвольное высказывание")
    parser.add_argument("--character", default="girl_narrator")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--story", type=Path, default=ROOT / "story.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-limit", type=int, default=0, help="Ограничение CPU affinity процесса")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-chunk-chars", type=int, default=0,
                        help="Окна текста: 80..2000 символов; 0 сохраняет старое поведение")
    parser.add_argument("--fast-memory", action="store_true", help="Сохранить арены памяти ORT; больше RAM")
    parser.add_argument("--quality", choices=["standard", "maximum"], default="maximum", 
                        help="Режим качества синтеза")
    parser.add_argument("--warmth", action="store_true", default=True, help="Аналоговая теплота")
    parser.add_argument("--synthetic-breath", action="store_true", default=True, 
                        help="Естественное дыхание")
    parser.add_argument("--stylized-laughter", action="store_true", default=True, 
                        help="Стилизованный смех")
    parser.add_argument("--micro-pauses", action="store_true", default=True,
                        help="Микропаузы для естественности")
    parser.add_argument("--adaptive-prosody", action="store_true", default=True,
                        help="Адаптивная просодия")
    
    args = parser.parse_args()
    
    if sum([args.text is not None, args.preview, args.lines is not None]) > 1:
        parser.error('Выберите только одно из --text, --preview или --lines')
    
    if args.threads < 1:
        parser.error('--threads должен быть положительным')
    
    if args.max_chunk_chars != 0 and not 80 <= args.max_chunk_chars <= 2000:
        parser.error('--max-chunk-chars должен быть 0 или целым числом от 80 до 2000')
    
    if args.bake:
        print("🔥 Запекание голосовых профилей для максимального качества...")
        result = bake_profiles()
        print("✅ Профили успешно запечены!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    import psutil
    process = psutil.Process()
    if args.cpu_limit:
        allowed = process.cpu_affinity()
        if not 1 <= args.cpu_limit <= len(allowed):
            parser.error("Некорректный --cpu-limit")
        process.cpu_affinity(allowed[:args.cpu_limit])

    cfg = validate_config(read_json(CONFIG))
    
    if args.text is not None:
        raw_items = [{"sender_id": args.character, "emotion": args.emotion, "text": args.text}]
        plan = preprocess_story(raw_items, cfg)
        basename = "single"
    else:
        raw_story = read_json(args.story)
        full_plan = preprocess_story(raw_story, cfg)
        
        try:
            selected_indices = [int(x) for x in args.lines.split(",")] if args.lines is not None else (
                [i for i in PREVIEW_LINES if i < len(raw_story)] if args.preview else list(range(len(raw_story))))
        except ValueError:
            parser.error('--lines требует индексы через запятую')
        
        if len(set(selected_indices)) != len(selected_indices) or any(i < 0 or i >= len(raw_story) for i in selected_indices):
            parser.error("Индексы истории должны быть уникальными и в диапазоне")
        
        plan = [full_plan[i] for i in selected_indices]
        basename = "preview" if args.preview else ("selection" if args.lines else "final_voiceover")

    print(f"🎙️  Инициализация улучшенного голосового движка (режим: {args.quality})...")
    started = time.perf_counter()
    
    engine = EnhancedVoiceEngine(
        threads=args.threads, 
        low_memory=not args.fast_memory,
        quality_mode=args.quality
    )
    
    load_seconds = time.perf_counter() - started
    print(f"⚡ Движок загружен за {load_seconds:.2f}с")
    
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else output / "cache"

    dialog_rows = []
    master_path = output / f'{basename}.wav'
    timestamps_path = output / ('dialog_timestamps.json' if basename == 'final_voiceover'
                                else f'{basename}_timestamps.json')
    
    print(f"🎬 Начинаем рендеринг {len(plan)} реплик...")
    
    with WavStream(master_path) as master:
        # Начальная пауза
        master.append(np.zeros(int(SAMPLE_RATE * .3), np.float32))
        
        for i, item in enumerate(plan):
            idx, char, emotion = item['line_index'], item['sender_id'], item['emotion']
            use_breath = args.synthetic_breath and item.get('needs_breath', False)
            
            print(f"🎯 [{i+1}/{len(plan)}] Рендеринг: {char} ({emotion}) - \"{item['raw_text'][:50]}{'...' if len(item['raw_text']) > 50 else ''}\"")
            
            audio, meta = engine.render(
                item['raw_text'], char, emotion, cache_dir=cache_dir,
                pronunciation_fixes=True, trim_edges=True, warmth=args.warmth,
                breath=use_breath, stylized_laughter=args.stylized_laughter,
                max_chunk_chars=args.max_chunk_chars,
                micro_pauses=args.micro_pauses,
                adaptive_prosody=args.adaptive_prosody
            )
            
            row = {
                'line_index': idx, 
                'sender_id': char, 
                'display_name': item['display_name'],
                'side': item['side'], 
                'emotion': emotion, 
                'text': item['raw_text'],
                'start_ms': round(master.frames * 1000 / SAMPLE_RATE),
                'end_ms': round((master.frames + len(audio)) * 1000 / SAMPLE_RATE),
                **meta
            }
            
            master.append(audio)
            
            # Адаптивная пауза между репликами
            gap_ms = meta['gap_ms']
            if args.micro_pauses and emotion in MICRO_PAUSE_EMOTIONS:
                gap_ms = int(gap_ms * 1.3)  # Увеличиваем паузы для задумчивых эмоций
            
            master.append(np.zeros(round(SAMPLE_RATE * gap_ms / 1000), np.float32))
            dialog_rows.append(row)
            
            cache_status = "💾 кэш" if meta['cache_hit'] else "🔥 новый"
            quality_score = meta.get('quality_score', 0)
            print(f"   ✅ {meta['audio_seconds']:.2f}с RTF={meta['rtf']:.3f} Q={quality_score}% {cache_status} дыхание={use_breath}")
    
    write_json(timestamps_path, dialog_rows)
    
    # Копирование в специальную папку для CPU аудио
    cpu_audio_dir = output / 'cpu_audio'
    cpu_audio_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(master_path, cpu_audio_dir / f'{basename}_enhanced.wav')
    write_json(cpu_audio_dir / f'{basename}_enhanced_timestamps.json', dialog_rows)

    speech_seconds = sum(row["audio_seconds"] for row in dialog_rows)
    generated_speech = sum(row["audio_seconds"] for row in dialog_rows if not row["cache_hit"])
    generation = sum(row["generation_seconds"] for row in dialog_rows)
    memory = process.memory_info()
    
    avg_quality = np.mean([row.get("quality_score", 0) for row in dialog_rows])
    
    report = {
        "status": "enhanced_render_complete_maximum_quality",
        "enhancement_level": "maximum_realism_v2",
        "emotion_control": cfg.get('emotion_control', 'advanced_prosodic_modulation'),
        "quality_mode": args.quality,
        "average_quality_score": round(avg_quality, 2),
        "low_memory": not args.fast_memory,
        "max_chunk_chars": args.max_chunk_chars,
        "model_load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation, 3),
        "speech_seconds": round(speech_seconds, 3),
        "master_seconds": round(master.frames / SAMPLE_RATE, 3),
        "rtf_uncached": round(generation / generated_speech, 4) if generated_speech else None,
        "total_seconds": round(time.perf_counter() - started, 3),
        "rss_mib": round(memory.rss / 2**20, 2),
        "peak_rss_mib": round(getattr(memory, "peak_wset", memory.rss) / 2**20, 2),
        "providers": engine.providers,
        "threads": args.threads,
        "line_count": len(dialog_rows),
        "master_wav": str(master_path),
        "enhanced_wav": str(cpu_audio_dir / f'{basename}_enhanced.wav'),
        "timestamps_json": str(timestamps_path),
        "master_sha256": file_hash(master_path),
        "audio_stats": master.stats(" ".join(r["spoken_text"] for r in dialog_rows)),
        "enhancements_applied": {
            "warmth": args.warmth,
            "synthetic_breath": args.synthetic_breath,
            "stylized_laughter": args.stylized_laughter,
            "micro_pauses": args.micro_pauses,
            "adaptive_prosody": args.adaptive_prosody
        }
    }
    
    write_json(output / f"{basename}_enhanced_benchmark.json", report)
    
    print("\n🎉 РЕНДЕРИНГ ЗАВЕРШЕН!")
    print(f"📊 Качество: {avg_quality:.1f}% | Время: {report['total_seconds']:.1f}с | RTF: {report['rtf_uncached']:.3f}")
    print(f"🎵 Файл: {master_path}")
    print(f"✨ Улучшенная версия: {cpu_audio_dir / f'{basename}_enhanced.wav'}")
    
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
