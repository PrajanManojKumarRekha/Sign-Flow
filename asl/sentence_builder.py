"""Turns a stream of detected gestures into text."""

import logging

from asl.config import BACKSPACE, DEBOUNCE_TIME

logger = logging.getLogger(__name__)


class SentenceBuilder:
    """Accumulates detected letters, ignoring repeats within a debounce window."""

    def __init__(self, debounce_time: float = DEBOUNCE_TIME):
        self.debounce_time = debounce_time
        self.text = ""
        self._last_gesture = None
        self._last_time = 0.0

    def add_character(self, char: str, current_time: float) -> bool:
        """Apply a gesture. Returns False if it was suppressed by debouncing."""
        if char == self._last_gesture and (current_time - self._last_time) < self.debounce_time:
            return False

        if char == BACKSPACE:
            self.text = self.text[:-1]
        else:
            self.text += char
        logger.info("Gesture %r -> %r", char, self.text)

        self._last_gesture = char
        self._last_time = current_time
        return True

    def get_text(self) -> str:
        return self.text if self.text else "[Empty]"

    def clear(self) -> None:
        self.text = ""
        self._last_gesture = None
        logger.info("Sentence cleared")
