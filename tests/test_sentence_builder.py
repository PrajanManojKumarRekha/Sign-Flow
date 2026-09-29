from asl.sentence_builder import SentenceBuilder


def test_appends_letters():
    b = SentenceBuilder(debounce_time=1.0)
    assert b.add_character("H", 0)
    assert b.add_character("I", 0.1)
    assert b.text == "HI"


def test_debounces_repeated_gesture():
    b = SentenceBuilder(debounce_time=1.0)
    assert b.add_character("A", 0)
    assert not b.add_character("A", 0.5)
    assert b.add_character("A", 1.5)
    assert b.text == "AA"


def test_backspace_and_empty():
    b = SentenceBuilder(debounce_time=1.0)
    assert b.get_text() == "[Empty]"
    b.add_character("backspace", 0)
    assert b.text == ""
    b.add_character("A", 1)
    b.add_character("backspace", 2)
    assert b.text == ""


def test_clear_resets_debounce():
    b = SentenceBuilder(debounce_time=1.0)
    b.add_character("A", 0)
    b.clear()
    assert b.add_character("A", 0.1)
