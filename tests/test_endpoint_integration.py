"""handle_audio sends end-of-audio when the short-burst endpointer fires and Home Assistant's VAD has not started."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from aioesphomeapi.api_pb2 import VoiceAssistantAudio

from linux_voice_assistant.endpointer import ShortBurstEndpointer
from linux_voice_assistant.satellite import VoiceSatelliteProtocol


def block(rms: int) -> bytes:
    return np.full(1024, rms, dtype="<i2").tobytes()


def _satellite() -> VoiceSatelliteProtocol:
    s = VoiceSatelliteProtocol.__new__(VoiceSatelliteProtocol)
    s.state = SimpleNamespace(muted=False, follow_up_timeout=5, listen_timeout=0)  # type: ignore[assignment]
    s.send_messages = MagicMock()  # type: ignore[method-assign]
    s._is_streaming_audio = True
    s._vad_started = False
    s._endpointer = ShortBurstEndpointer()
    s._ep_streaming = False
    return s


def _ends(s) -> int:
    return sum(1 for c in s.send_messages.call_args_list for m in c.args[0] if isinstance(m, VoiceAssistantAudio) and m.end)


def _speak_a_short_word(s, then_quiet=3.0):
    for rms, secs in [(30, 1.0), (700, 0.2), (30, then_quiet)]:
        for _ in range(int(round(secs / 0.064))):
            s.handle_audio(block(rms))


def test_short_word_without_ha_vad_ends_the_stream_once():
    s = _satellite()
    _speak_a_short_word(s)
    assert _ends(s) == 1
    assert not s._is_streaming_audio
    # audio before the end was streamed normally
    data_msgs = [m for c in s.send_messages.call_args_list for m in c.args[0] if isinstance(m, VoiceAssistantAudio) and not m.end]
    assert len(data_msgs) > 15


def test_nothing_is_sent_after_the_end():
    s = _satellite()
    _speak_a_short_word(s)
    sent = s.send_messages.call_count
    for _ in range(20):
        s.handle_audio(block(30))
    assert s.send_messages.call_count == sent


def test_ha_vad_started_means_no_end():
    s = _satellite()
    s._vad_started = False
    for rms, secs in [(30, 0.5), (700, 0.2)]:
        for _ in range(int(round(secs / 0.064))):
            s.handle_audio(block(rms))
    s._vad_started = True  # HA noticed the speech
    for _ in range(60):
        s.handle_audio(block(30))
    assert _ends(s) == 0 and s._is_streaming_audio


def test_new_stream_rearms_and_clears_the_vad_flag():
    s = _satellite()
    _speak_a_short_word(s)
    assert _ends(s) == 1
    s.send_messages.reset_mock()
    s._vad_started = True  # left over from the previous run
    s._is_streaming_audio = True  # a new stream opens (e.g. the retry)
    _speak_a_short_word(s)
    assert _ends(s) == 1


def test_muted_sends_nothing():
    s = _satellite()
    s.state.muted = True
    _speak_a_short_word(s)
    assert s.send_messages.call_count == 0


def test_follow_up_stream_ends_after_five_seconds_without_speech():
    s = _satellite()
    s._followup_stream = True
    for _ in range(int(round(8.0 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 1
    assert not s._is_streaming_audio
    assert not s._followup_stream  # one-shot: the next stream is a normal one


def test_normal_stream_has_no_silence_timeout():
    s = _satellite()
    for _ in range(int(round(8.0 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 0


def test_follow_up_timeout_comes_from_the_setting():
    s = _satellite()
    s.state.follow_up_timeout = 3
    s._followup_stream = True
    for _ in range(int(round(4.5 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 1


def test_follow_up_timeout_zero_leaves_it_to_home_assistant():
    s = _satellite()
    s.state.follow_up_timeout = 0
    s._followup_stream = True
    for _ in range(int(round(8.0 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 0


def test_listen_timeout_ends_a_silent_wake_word_stream():
    s = _satellite()
    s.state.listen_timeout = 3
    s._followup_stream = False
    for _ in range(int(round(4.5 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 1


def test_listen_timeout_zero_leaves_wake_word_stream_to_home_assistant():
    s = _satellite()
    s.state.listen_timeout = 0
    s._followup_stream = False
    for _ in range(int(round(8.0 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 0


def test_follow_up_keeps_its_own_timeout():
    s = _satellite()
    s.state.listen_timeout = 3
    s.state.follow_up_timeout = 6
    s._followup_stream = True
    for _ in range(int(round(4.5 / 0.064))):
        s.handle_audio(block(30))
    assert _ends(s) == 0
