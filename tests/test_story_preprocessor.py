import pytest
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import story_preprocessor as prep


def test_normalize_basic_sentence():
    assert prep.normalize_text("Hello, world!") == "Hello, world!"


def test_normalize_trailing_clause_preserved():
    assert prep.normalize_text("Wait! Keep this final clause") == "Wait! Keep this final clause"


def test_normalize_sigh_removal():
    assert prep.normalize_text("Sigh, we used to grill BBQ together...") == "we used to grill barbecue together..."
    assert prep.normalize_text("[sigh] That was rough.") == "That was rough."
    assert prep.normalize_text("(sigh) What a day.") == "What a day."
    # Sight and Sighing must not be corrupted
    assert prep.normalize_text("Sight is precious.") == "Sight is precious."
    assert prep.normalize_text("Sighing deeply is natural.") == "Sighing deeply is natural."


def test_normalize_numbers_and_currency():
    assert prep.normalize_text("It costs $12.50 today.") == "It costs twelve dollars and fifty cents today."
    assert prep.normalize_text("Only £1 left.") == "Only one pound left."
    assert prep.normalize_text("He had €100 in cash.") == "He had one hundred euros in cash."


def test_normalize_time_and_abbreviations():
    assert prep.normalize_text("Meeting at 5:05 PM.") == "Meeting at five oh five P M."
    assert prep.normalize_text("Check CPU, API, and USB status.") == "Check C P U, A P I, and U S B status."
    assert prep.normalize_text("Visit the USA and UK.") == "Visit the U S A and U K."


def test_normalize_laughter_and_punctuation():
    assert prep.normalize_text("Hahaha! That was wild?!") == "Ha Ha Ha! That was wild?"


def test_unsupported_inputs_raise():
    with pytest.raises(ValueError):
        prep.normalize_text("")
    with pytest.raises(ValueError):
        prep.normalize_text("   ")
    with pytest.raises(ValueError):
        prep.normalize_text("[cry] please stop")
    with pytest.raises(ValueError):
        prep.normalize_text("Привет мир")  # Non-English fails closed


def test_hooray_pronunciation_override():
    class DummyTokenizer:
        def phonemize(self, text, lang):
            return "hˈɔːɹeɪ, wiː mˈeɪd ɪt!"
        def known(self, val):
            return val
    
    out, changed, fixes = prep.pronunciation_override(DummyTokenizer(), "Hooray, we made it!")
    assert changed is True
    assert out == "həɹˈeɪ, wiː mˈeɪd ɪt!"
    assert fixes[0]["word"] == "Hooray"


def test_dynamic_laughter_syllable_scaling():
    class LaughTokenizer:
        def phonemize(self, text, lang):
            count = len(re.findall(r'\bHa\b', text, re.I))
            return " ".join(["hˈɑː"] * count) + "! The show is on"
        def known(self, val):
            return val

    out2, ch2, fixes2 = prep.pronunciation_override(LaughTokenizer(), "Ha Ha! The show is on")
    assert ch2 and out2.startswith("hˈʌhʌ,") and fixes2[0]["syllables"] == 2

    out3, ch3, fixes3 = prep.pronunciation_override(LaughTokenizer(), "Ha Ha Ha! The show is on")
    assert ch3 and out3.startswith("hˈʌhʌhʌ,") and fixes3[0]["syllables"] == 3

    out5, ch5, fixes5 = prep.pronunciation_override(LaughTokenizer(), "Ha Ha Ha Ha Ha! The show is on")
    assert ch5 and out5.startswith("hˈʌhʌhʌhʌhʌ,") and fixes5[0]["syllables"] == 5


def test_preprocess_story_structure():
    config = {
        "characters": {
            "girl_narrator": {"voice": "af_heart", "speed": 0.97, "display_name": "Jess", "side": "right"},
            "karen": {"voice": "af_sarah", "speed": 0.96, "display_name": "Karen", "side": "left"}
        },
        "emotions": {
            "neutral": {"speed": 1.0, "gap_ms": 380},
            "anger": {"speed": 1.02, "gap_ms": 300}
        }
    }
    raw_story = [
        {"sender_id": "girl_narrator", "emotion": "neutral", "text": "Welcome everyone."},
        {"sender_id": "karen", "emotion": "anger", "text": "Who parked here?!"}
    ]
    plan = prep.preprocess_story(raw_story, config)
    assert len(plan) == 2
    assert plan[0]["display_name"] == "Jess" and plan[0]["side"] == "right"
    assert plan[0]["needs_breath"] is False  # First line never has breath
    assert plan[1]["needs_breath"] is True   # Anger on second line triggers breath
    assert plan[1]["gap_ms"] == 300
    assert plan[1]["speed"] == round(0.96 * 1.02, 4)


@pytest.mark.parametrize('text', ['Hi\x7f', 'Hi\x85', 'Hi\u200b', 'Hi\ud800', '...', 'a' * 2001,
                                '$1,23', '$12.345', 'Meet at 13:00 PM.', '[unknown] Hi.',
                                '(whisper) Hello.', 'Hi (sigh).'])
def test_boundary_inputs_rejected(text):
    with pytest.raises(ValueError):
        prep.normalize_text(text)


@pytest.mark.parametrize('text', ['Hello\nworld\tand friends.', 'a' * 2000,
                                'Hooray! Keep the unpunctuated ending',
                                'Doctor Smith paid $1,234.56 at 12:00 AM.'])
def test_valid_boundaries_and_idempotence(text):
    clean = prep.normalize_text(text)
    assert prep.normalize_text(clean) == clean


@pytest.mark.parametrize('story', [[], {}, [None], [{'sender_id': 'unknown', 'text': 'Hi.'}],
                                 [{'sender_id': 'a', 'emotion': 'unknown', 'text': 'Hi.'}]])
def test_invalid_story_rejected(story):
    cfg = {'characters': {'a': {'speed': 1}}, 'emotions': {'neutral': {'speed': 1}}}
    with pytest.raises(ValueError):
        prep.preprocess_story(story, cfg)


def test_long_stylized_laughter_preserves_syllables_and_tail():
    class Tokenizer:
        def phonemize(self, text, lang):
            return ' '.join(['hˈɑː'] * 10) + '! kiːp ðə tˈeɪl'
        def known(self, val):
            return val
    out, applied, fixes = prep.pronunciation_override(Tokenizer(), ' '.join(['Ha'] * 10) + '! Keep the tail')
    assert applied and out.startswith('hˈʌ' + 'hʌ' * 9 + ',')
    assert out.endswith('kiːp ðə tˈeɪl') and fixes[0]['syllables'] == 10


@pytest.mark.parametrize('limit', [0, 80, 160, 2000])
def test_split_preserves_words_punctuation_and_final_clause(limit):
    text = prep.normalize_text(('Please keep every word in this sentence ' * 40) + 'this is the ending')
    chunks = prep.split_synthesis_text(text, limit)
    assert ' '.join(chunks) == text
    assert chunks[-1].endswith('this is the ending')
    assert all(chunks)
    if limit:
        assert all(len(chunk) <= limit for chunk in chunks)
    else:
        assert chunks == [text]


@pytest.mark.parametrize('limit', [-1, 1, 79, 2001, True, False, 80.0, '80', None])
def test_split_invalid_limit_rejected(limit):
    with pytest.raises(ValueError, match='max_chunk_chars'):
        prep.split_synthesis_text('Hello.', limit)


@pytest.mark.parametrize('text', ['', '  ', None])
def test_split_empty_input_rejected(text):
    with pytest.raises(ValueError, match='empty'):
        prep.split_synthesis_text(text, 80)


@pytest.mark.parametrize('punctuation', ['.', '!', '?', '."', ','])
def test_split_prefers_sentence_or_clause_over_later_space(punctuation):
    first = 'We must preserve every spoken word in this phrase' + punctuation
    text = first + ' And now keep all the remaining words without losing the ending'
    chunks = prep.split_synthesis_text(text, 80)
    assert chunks[0] == first
    assert ' '.join(chunks) == text


def test_split_sentence_has_priority_over_later_clause():
    first = 'We must preserve every spoken word in this phrase.'
    text = first + ' A short clause, and the rest of this sentence must stay complete'
    assert prep.split_synthesis_text(text, 80)[0] == first


@pytest.mark.parametrize('text', ['a' * 81, 'short ' + 'a' * 81, 'a' * 81 + ' short'])
def test_split_refuses_to_cut_long_words(text):
    with pytest.raises(ValueError):
        prep.split_synthesis_text(text, 80)
    assert prep.split_synthesis_text(text, 0) == [text]


def test_split_exact_boundary_and_normalization_expansion():
    assert prep.split_synthesis_text('a' * 80, 80) == ['a' * 80]
    clean = prep.normalize_text('CPU ' * 400)
    assert len(clean) > 2000
    assert ' '.join(prep.split_synthesis_text(clean, 160)) == clean
