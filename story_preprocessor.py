"""Linguistic preprocessor for story text.

Layer 2 of the 3-layer architecture:
  1. Story (story.json) -> 2. Preprocessor (story_preprocessor.py) -> 3. Voice Engine (cpu_voice_engine.py)

Responsibilities:
  - Text normalization: numbers, currencies ($/£/€), 12-hour clock times, abbreviations, units, contractions.
  - Safe removal of unsupported stage directions ([sigh], (sigh), leading sigh).
  - Phonetic dictionary overrides (Hooray -> həɹˈeɪ) preserving complete utterances.
  - Emotional pacing and dramatic breath transition detection.
  - Generates a validated execution plan ready for acoustic rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "cpu_voices.json"
BREATH_EMOTIONS = {"anger", "fear", "sadness", "surprise"}


def integer_words(value: int) -> str:
    """Deterministic en-US cardinal formatter; no external dependencies."""
    n = int(value)
    if not 0 <= n < 10**12:
        raise ValueError("Number outside supported range")
    ones = ('zero one two three four five six seven eight nine ten eleven twelve '
            'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    tens = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ('-' + ones[n % 10] if n % 10 else '')
    for scale, name in [(10**9, 'billion'), (10**6, 'million'), (1000, 'thousand'), (100, 'hundred')]:
        if n >= scale:
            return integer_words(n // scale) + ' ' + name + (' ' + integer_words(n % scale) if n % scale else '')
    return str(n)


def money_words(match: re.Match) -> str:
    symbol, amount = match.group(1), match.group(2).replace(',', '')
    whole, _, fraction = amount.partition('.')
    major, minor = {'$': ('dollar', 'cent'), '£': ('pound', 'penny'), '€': ('euro', 'cent')}[symbol]
    n, cents = int(whole), int(fraction.ljust(2, '0') or '0')
    result = integer_words(n) + ' ' + major + ('' if n == 1 else 's')
    if cents:
        minor_plural = 'pence' if minor == 'penny' else minor + 's'
        result += ' and ' + integer_words(cents) + ' ' + (minor if cents == 1 else minor_plural)
    return result


def clock_words(match: re.Match) -> str:
    hour, minute, period = int(match.group(1)), int(match.group(2)), match.group(3)
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        raise ValueError('Invalid 12-hour clock time')
    tail = (" o'clock" if minute == 0 else ' oh ' + integer_words(minute) if minute < 10
            else ' ' + integer_words(minute))
    return integer_words(hour) + tail + ' ' + ' '.join(period.upper().replace('.', ''))


def normalize_text(text: str) -> str:
    """Normalize en-US dialogue text without phonetic distortion or dropped clauses."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text must be a non-empty string')
    if len(text) > 2000:
        raise ValueError('Text exceeds 2000 characters; split into dialogue lines')
    if any(unicodedata.category(c) in {'Cf', 'Cs'} for c in text):
        raise ValueError('Unsupported invisible or surrogate character in text')
    if any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text):
        raise ValueError('This configured speech pipeline supports English text only')
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('\u2026', '...')
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u00a0', ' ')
    if re.search(r'[\u0400-\u052f\u3040-\u30ff\u3400-\u9fff]', text):
        raise ValueError('This configured speech pipeline supports English text only')
    if any(unicodedata.category(c) == 'Cc' and c not in '\t\r\n' for c in text):
        raise ValueError('Unsupported control character in text')
    if re.search(r'\[(?:laugh\w*|chuckl\w*|sob\w*|cry\w*|gasp\w*|scream\w*|whisper\w*)\]', text, re.I):
        raise ValueError('Non-speech direction is unsupported; a real recorded vocalization is required')
    
    # Strip leading stage-direction sigh safely without corrupting Sight/Sighing/Sighs
    text = re.sub(r'^\s*(?:\[sigh\]|\(sigh\)|sigh\b(?=\s*[,.:;!?-]|\s*$))\s*[,.:;!?-]*\s*', '', text, flags=re.I)
    if re.search(r'\[[^\]]*\]|\((?:sigh|laugh\w*|sob\w*|cry\w*|gasp\w*|scream\w*|whisper\w*)\)', text, re.I):
        raise ValueError('Unsupported stage direction; provide spoken text or a recorded vocalization')
    for amount in re.findall(r'[$£€]\s*([\d,.]+)', text):
        if not re.fullmatch(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?', amount.rstrip('.,')):
            raise ValueError('Ambiguous currency amount')
    text = re.sub(r'(?<!\w)([$£€])((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.[0-9]{1,2})?)(?!\d|[.,]\d)', money_words, text)
    text = re.sub(r'\b(\d{1,2}):(\d{2})\s*([AP]\.?M)\b', clock_words, text, flags=re.I)
    text = re.sub(r'\b(\d+(?:\.\d+)?)%', lambda m: m[1] + ' percent', text)
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*°\s*([CF])\b',
                  lambda m: m[1] + ' degrees ' + ('Celsius' if m[2] == 'C' else 'Fahrenheit'), text)
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*(kg|km|cm|mm)\b',
                  lambda m: m[1] + ' ' + {'kg': 'kilograms', 'km': 'kilometers', 'cm': 'centimeters', 'mm': 'millimeters'}[m[2]], text)
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms)\.(?=\s+[A-Z][a-z])',
                  lambda m: {'Dr': 'Doctor', 'Mr': 'Mister', 'Mrs': 'Missus', 'Ms': 'Miz'}[m[1]], text)
    expansions = {'BBQ': 'barbecue', 'CPU': 'C P U', 'GPU': 'G P U', 'API': 'A P I',
                  'USB': 'U S B', 'AI': 'A I', 'URL': 'U R L', 'HTTP': 'H T T P',
                  'HTTPS': 'H T T P S', 'TLS': 'T L S', 'PDF': 'P D F', 'ID': 'I D',
                  'SMS': 'S M S', 'GPS': 'G P S', 'FBI': 'F B I', 'UK': 'U K', 'USA': 'U S A'}
    text = re.sub(r'\b(?:' + '|'.join(sorted(expansions, key=len, reverse=True)) + r')\b',
                  lambda m: expansions[m.group()], text)
    text = re.sub(r'\b(?:ha){2,}\b', lambda m: ' '.join(['Ha'] * (len(m.group()) // 2)), text, flags=re.I)
    text = re.sub(r'[!?]{2,}', lambda m: '?' if '?' in m.group() else '!', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text or not re.search(r'[A-Za-z0-9]', text):
        raise ValueError('Text contains no supported speech')
    return text


def split_synthesis_text(text: str, max_chars: int = 0) -> List[str]:
    """Split already-normalized text losslessly; 0 keeps legacy synthesis.

    Prefer sentence/clause boundaries in the latter half of the window, then
    whitespace. Never split a word or add punctuation. This bounds text windows,
    NOT phoneme counts or total process RAM. Do not normalize expanded text again.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or (max_chars != 0 and not 80 <= max_chars <= 2000):
        raise ValueError('max_chunk_chars must be 0 or an integer from 80 to 2000')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Cannot split empty text')
    if not max_chars or len(text) <= max_chars:
        return [text]
    parts, remaining = [], text
    while len(remaining) > max_chars:
        boundaries = [m for m in re.finditer(r'\s+', remaining) if 0 < m.start() <= max_chars]
        if not boundaries:
            raise ValueError('A word exceeds max_chunk_chars; refusing to truncate it')
        preferred = [m for m in boundaries if m.start() >= max_chars // 2]
        sentence = [m for m in preferred if remaining[:m.start()].rstrip('\"\')').endswith(('.', '!', '?'))]
        clause = [m for m in preferred if remaining[:m.start()].rstrip('\"\')').endswith((',', ';', ':'))]
        cut = (sentence or clause or boundaries)[-1]
        parts.append(remaining[:cut.start()])
        remaining = remaining[cut.end():]
    if remaining:
        parts.append(remaining)
    if ' '.join(parts).split() != text.split() or any(len(p) > max_chars for p in parts):
        raise ValueError('Synthesis split changed words or exceeded its limit')
    return parts


def pronunciation_override(tokenizer_or_tts: Any, text: str, language: str = 'en-us', include_ew: bool = False, include_laughter: bool = True) -> Tuple[str, bool, List[Dict[str, Any]]]:
    """Word-level fixes; optional stylized laughter is NOT a real vocal performance."""
    fixes = []
    has_hooray = bool(re.search(r'\bHooray\b', text, re.I))
    has_ew = bool(include_ew and re.search(r'\bEw\b', text, re.I))
    has_laugh = bool(include_laughter and re.search(r'\b(?:ha)(?:\s+ha)+\b|\b(?:ha){2,}\b|\b(?:he){2,}\b', text, re.I))
    if not has_hooray and not has_ew and not has_laugh:
        return text, False, fixes

    tokenizer = getattr(tokenizer_or_tts, 'tokenizer', tokenizer_or_tts)
    phonemes = tokenizer.phonemize(text, language)
    applied = False

    if has_ew:
        if re.search(r'\byou\b', text, re.I):
            fixes.append({'word': 'Ew', 'status': 'not_applied_ambiguity_with_you'})
        else:
            fixes.append({'word': 'Ew', 'status': 'not_applied_default'})

    if has_hooray:
        word, old, new = 'Hooray', 'hˈɔːɹeɪ', 'həɹˈeɪ'
        count = len(re.findall(r'\b' + word + r'\b', text, re.I))
        pattern = r'(?<!\S)' + re.escape(old) + r'(?=[\s,.!?;:]|$)'
        if count and len(re.findall(pattern, phonemes)) == count:
            revised = re.sub(pattern, new, phonemes)
            if hasattr(tokenizer, 'known') and tokenizer.known(revised) != revised:
                raise ValueError('Pronunciation override contains unsupported symbols')
            fixes.append({'word': word, 'before': old, 'after': new, 'count': count})
            phonemes = revised
            applied = True

    if has_laugh:
        # Experimental phonetic approximation, NOT natural recorded laughter.
        # Preserve the full syllable count instead of silently truncating it.
        pattern_ha = r'(?<!\S)(?:h[ˈˌ]?ɑː\s*){2,}[!?,.]*(?=[\s]|$)'
        pattern_he = r'(?<!\S)(?:h[ˈˌ]?ɛ\s*){2,}[!?,.]*(?=[\s]|$)'

        def laugh_replacer_ha(m):
            s = m.group(0)
            n = len(re.findall(r'h[ˈˌ]?ɑː', s))
            n = max(2, n)
            return 'hˈʌ' + 'hʌ' * (n - 1) + ','

        def laugh_replacer_he(m):
            s = m.group(0)
            n = len(re.findall(r'h[ˈˌ]?ɛ', s))
            n = max(2, n)
            return 'hˈɛ' + 'hɛ' * (n - 1) + ','

        if re.search(pattern_ha, phonemes):
            revised = re.sub(pattern_ha, laugh_replacer_ha, phonemes)
            n_count = len(re.findall(r'h[ˈˌ]?ɑː', re.search(pattern_ha, phonemes).group(0)))
            fixes.append({'word': 'Laughter', 'syllables': n_count, 'style': 'dynamic_caret'})
            phonemes = revised
            applied = True
        elif re.search(pattern_he, phonemes):
            revised = re.sub(pattern_he, laugh_replacer_he, phonemes)
            n_count = len(re.findall(r'h[ˈˌ]?ɛ', re.search(pattern_he, phonemes).group(0)))
            fixes.append({'word': 'Laughter', 'syllables': n_count, 'style': 'dynamic_eh'})
            phonemes = revised
            applied = True

    # If no substitution was made, leave orthography for the engine to phonemize.
    return (phonemes if applied else text), applied, fixes


def preprocess_story(story_data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform raw story lines into fully validated execution plan items."""
    if not isinstance(story_data, list) or not story_data:
        raise ValueError('Story must be a non-empty list')
    chars_cfg = config.get("characters", {})
    emotions_cfg = config.get("emotions", {})

    plan = []
    for idx, item in enumerate(story_data):
        if not isinstance(item, dict):
            raise ValueError(f'Line {idx}: expected an object')
        char = item.get("sender_id")
        if not char or char not in chars_cfg:
            raise ValueError(f"Line {idx}: Unknown character '{char}'")
        
        emotion = item.get("emotion", "neutral")
        if emotion not in emotions_cfg:
            raise ValueError(f"Line {idx}: Unknown emotion '{emotion}'")
        
        raw_text = item.get("text", "")
        clean_text = normalize_text(raw_text)
        
        char_info = chars_cfg[char]
        pacing_info = emotions_cfg[emotion]
        base_speed = float(char_info.get("speed", 1.0))
        emotion_speed = float(pacing_info.get("speed", 1.0))
        final_speed = float(max(0.5, min(2.0, base_speed * emotion_speed)))
        
        # Dramatic breath triggered before emotional phrases (except first line)
        needs_breath = (emotion in BREATH_EMOTIONS) and (idx > 0)
        
        # Display name & UI side for video integration
        display_name = item.get("display_name", char_info.get("display_name", char.replace("_", " ").title()))
        side = item.get("side", char_info.get("side", "right" if char == "girl_narrator" else "left"))
        
        plan_item = {
            "line_index": idx,
            "sender_id": char,
            "display_name": display_name,
            "side": side,
            "emotion": emotion,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "speed": round(final_speed, 4),
            "gap_ms": pacing_info.get("gap_ms", 380),
            "needs_breath": needs_breath,
            "voice_target": ("+".join(f"{v}:{w}" for v, w in sorted(char_info["voice_blend"].items()))
                             if "voice_blend" in char_info else char_info.get("voice", char))
        }
        plan.append(plan_item)
    return plan


def main():
    parser = argparse.ArgumentParser(description="Preprocess story dialogue into execution plan.")
    parser.add_argument("story", type=Path, nargs="?", default=ROOT / "story.json")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "output" / "execution_plan.json")
    parser.add_argument("-c", "--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    
    with open(args.story, "r", encoding="utf-8") as f:
        story = json.load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    plan = preprocess_story(story, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"[✓] Preprocessed {len(plan)} story lines -> {args.output}")


if __name__ == "__main__":
    main()
