"""Tests for the wake word loudness log (wake_metrics.LevelHistory, wake_metrics.format_metrics)."""

import numpy as np

from linux_voice_assistant.wake_metrics import LevelHistory, format_metrics


def _tone(seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_levels_follow_loudness_oldest_first():
    h = LevelHistory(frame_ms=20, seconds=1.0)
    h.add(_tone(0.5, 0.01))
    h.add(_tone(0.5, 0.5))
    levels = h.levels()
    assert len(levels) == 50
    assert levels[0] < -40 and levels[-1] > -10          # quiet room first, the loud word last


def test_history_keeps_only_the_last_seconds():
    h = LevelHistory(frame_ms=20, seconds=0.2)
    h.add(_tone(1.0, 0.5))
    h.add(_tone(0.2, 0.01))
    assert len(h.levels()) == 10 and max(h.levels()) < -40


def test_blocks_not_aligned_to_frames_are_carried_over():
    h = LevelHistory(frame_ms=20, seconds=1.0)
    for _ in range(10):
        h.add(_tone(0.03, 0.1))                            # 480 samples, frames are 320
    assert len(h.levels()) == 15


def test_silence_has_a_floor_not_minus_infinity():
    h = LevelHistory(frame_ms=20, seconds=0.1)
    h.add(np.zeros(1600, dtype=np.float32))
    assert h.levels() == [-120] * 5


def test_log_line_carries_word_probability_and_levels():
    line = format_metrics("Alexa", 0.91234, 20, [-60, -12])
    assert line == "Wake metrics: word=Alexa prob=0.912 frame_ms=20 levels=-60,-12"
    assert "prob=-" in format_metrics("Alexa", None, 20, [])


def test_snr_is_the_word_peak_over_the_median_right_before_it():
    from linux_voice_assistant.wake_metrics import wake_snr

    levels = [-94] * 50 + [-64] * 50 + [-30] + [-60] * 49          # silence of the own speaker, the room, then the word
    assert wake_snr(levels, 20) == 34.0                             # the silence 2 s back does not count as the room


def test_snr_needs_history_for_both_windows():
    from linux_voice_assistant.wake_metrics import wake_snr

    assert wake_snr([-30] * 60, 20) is None
