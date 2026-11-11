"""Loudness of the raw microphone around a wake word, logged to compare satellites that heard the same word.

Several satellites in neighbouring rooms hear one wake word and Home Assistant keeps the first one to report it, not the nearest.
This log is the measurement step before arbitration between satellites: each detection writes the level envelope of the last seconds
(raw microphone, before mic volume, auto gain and noise suppression), so the wake word's level against the room's noise can be
compared offline across satellites. Off unless --wake-metrics-seconds is set.
"""

from collections import deque
from typing import Deque, List, Optional

import numpy as np

_RATE = 16000
_FLOOR_DB = -120  # the level of digital silence: log10(0) has no value


class LevelHistory:
    """RMS level (dBFS, whole dB) of each `frame_ms` frame of the raw microphone over the last `seconds`."""

    def __init__(self, frame_ms: int, seconds: float) -> None:
        self._frame = max(1, _RATE * frame_ms // 1000)
        self._levels: Deque[int] = deque(maxlen=max(1, int(seconds * 1000 // frame_ms)))
        self._rest = np.zeros(0, dtype=np.float32)

    def add(self, samples: np.ndarray) -> None:
        """Add a block of mono float samples in [-1, 1]; a part shorter than a frame waits for the next block."""
        data = np.concatenate([self._rest, samples.astype(np.float32, copy=False)])
        whole = len(data) // self._frame * self._frame
        if whole:
            frames = data[:whole].reshape(-1, self._frame)
            rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
            for value in rms:
                self._levels.append(int(round(20 * np.log10(value))) if value > 0 else _FLOOR_DB)
        self._rest = data[whole:].copy()

    def levels(self) -> List[int]:
        """The levels, oldest first, never below the silence floor."""
        return [max(_FLOOR_DB, v) for v in self._levels]


def format_metrics(word: str, probability: Optional[float], frame_ms: int, levels: List[int]) -> str:
    """One log line per detection; `probability` is the wake word model's mean over its sliding window (None when unknown)."""
    prob = f"{probability:.3f}" if probability is not None else "-"
    return f"Wake metrics: word={word} prob={prob} frame_ms={frame_ms} levels={','.join(str(v) for v in levels)}"


DEFAULT_WORD_MS = 1000  # the wake word ends right before its detection: its loudest frame lies in this much before it
DEFAULT_NOISE_MS = 1000  # the room right before the word; earlier frames can hold the digital silence some microphones give while
# the satellite's own speaker plays (bedroom 2026-11-11: -94 dBFS instead of the radiator's -64)


def wake_level(levels: List[int], frame_ms: int, word_ms: int = DEFAULT_WORD_MS) -> Optional[float]:
    """The wake word's loudest frame (dBFS) in the last `word_ms`: unlike the SNR it does not drop when the room is loud (music in the
    bedroom on 2026-11-11 raised its noise 8 dB and gave the hallway its wake words); the satellites' microphones differ, so it needs
    each satellite's offset. None when the history is shorter than `word_ms`."""
    n_word = max(1, word_ms // frame_ms)
    return float(max(levels[-n_word:])) if len(levels) >= n_word else None


def wake_snr(levels: List[int], frame_ms: int, word_ms: int = DEFAULT_WORD_MS, noise_ms: int = DEFAULT_NOISE_MS) -> Optional[float]:
    """How much louder the wake word was than the room right before it (dB): the loudest frame of the last `word_ms` less the median
    of the `noise_ms` before them. None when the history is too short to hold both (the satellite just started)."""
    n_word, n_noise = max(1, word_ms // frame_ms), max(1, noise_ms // frame_ms)
    if len(levels) < n_word + n_noise:
        return None
    word = levels[-n_word:]
    before = sorted(levels[-(n_word + n_noise):-n_word])
    return float(max(word) - before[len(before) // 2])


SCORES = {"level": wake_level, "snr": wake_snr}  # the arbitration's score of a wake word (--arbitration-score): name -> fn(levels, frame_ms)
