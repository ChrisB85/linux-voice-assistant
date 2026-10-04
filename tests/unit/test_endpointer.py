"""LVA-side end-of-speech for short words that Home Assistant's VAD never notices (it needs 0.3 s of speech)."""

import numpy as np

from linux_voice_assistant.endpointer import ShortBurstEndpointer

CHUNK = 1024  # 64 ms at 16 kHz


def block(rms: float) -> bytes:
    # constant-amplitude block with the requested RMS (a square wave has rms == amplitude)
    return (np.full(CHUNK, int(rms), dtype="<i2")).tobytes()


def feed(ep, profile, ha_vad=False):
    """profile: list of (rms, seconds). Returns the stream time (s) at which end was requested, or None."""
    t = 0.0
    for rms, secs in profile:
        for _ in range(int(round(secs / 0.064))):
            t += 0.064
            if ep.feed(block(rms), ha_vad):
                return t
    return None


def test_short_burst_then_silence_requests_end_after_a_second():
    ep = ShortBurstEndpointer()
    t = feed(ep, [(30, 1.0), (700, 0.2), (30, 3.0)])
    assert t is not None
    assert 1.2 + 0.9 <= t <= 1.2 + 1.2        # burst ends at 1.2 s, +1 s of quiet


def test_pure_silence_never_ends():
    assert feed(ShortBurstEndpointer(), [(30, 20.0)]) is None


def test_burst_without_enough_trailing_silence_does_not_end():
    assert feed(ShortBurstEndpointer(), [(30, 1.0), (700, 0.2), (30, 0.8)]) is None


def test_home_assistant_vad_started_disables_it():
    assert feed(ShortBurstEndpointer(), [(30, 1.0), (700, 0.2), (30, 3.0)], ha_vad=True) is None


def test_pause_shorter_than_a_second_inside_speech_does_not_end():
    t = feed(ShortBurstEndpointer(), [(30, 0.5), (700, 0.3), (30, 0.7), (700, 0.4), (30, 0.5)])
    assert t is None


def test_burst_must_stand_out_from_a_noisy_floor():
    # steady 300 rms noise (vacuum cleaner): a 900 burst is only 3x, a 2500 burst is >6x
    assert feed(ShortBurstEndpointer(), [(300, 2.0), (900, 0.2), (300, 3.0)]) is None
    assert feed(ShortBurstEndpointer(), [(300, 2.0), (2500, 0.2), (300, 3.0)]) is not None


def test_quiet_blips_below_the_minimum_peak_are_ignored():
    assert feed(ShortBurstEndpointer(), [(5, 2.0), (120, 0.2), (5, 3.0)]) is None


def test_requests_end_only_once_and_reset_rearms():
    ep = ShortBurstEndpointer()
    assert feed(ep, [(30, 1.0), (700, 0.2), (30, 3.0)]) is not None
    assert feed(ep, [(30, 3.0)]) is None       # already requested for this stream
    ep.reset()
    assert feed(ep, [(30, 1.0), (700, 0.2), (30, 3.0)]) is not None


def test_sound_in_the_first_second_of_the_stream_is_ignored():
    # the listening chime / echo of the question right after the mic opens must not look like a spoken word
    assert feed(ShortBurstEndpointer(), [(30, 0.1), (700, 0.3), (30, 5.0)]) is None


def test_chime_then_real_word_still_ends_after_the_word():
    t = feed(ShortBurstEndpointer(), [(30, 0.1), (700, 0.3), (30, 1.7), (700, 0.2), (30, 3.0)])
    assert t is not None
    assert 2.3 + 0.9 <= t <= 2.3 + 1.2        # word ends at 2.3 s, +1 s of quiet


def test_no_speech_timeout_ends_a_silent_stream():
    # nobody answers (a chime at the start only): stop after no_speech_s instead of HA's 15 s
    t = feed(ShortBurstEndpointer(no_speech_s=5.0), [(30, 0.1), (2800, 0.4), (30, 12.0)])
    assert t is not None
    assert 4.9 <= t <= 5.2


def test_no_speech_timeout_is_off_by_default():
    assert feed(ShortBurstEndpointer(), [(30, 0.1), (2800, 0.4), (30, 12.0)]) is None


def test_no_speech_timeout_does_not_cut_a_word_in_progress():
    # speech at 4.8 s still going when 5 s passes: the burst rule decides, not the timeout
    assert feed(ShortBurstEndpointer(no_speech_s=5.0), [(30, 4.8), (700, 1.5)]) is None


def test_no_speech_timeout_leaves_the_short_word_rule_alone():
    t = feed(ShortBurstEndpointer(no_speech_s=5.0), [(30, 2.0), (700, 0.2), (30, 6.0)])
    assert t is not None
    assert 2.2 + 0.9 <= t <= 2.2 + 1.2
