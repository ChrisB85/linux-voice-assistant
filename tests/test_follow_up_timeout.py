"""A follow-up listening (the assistant asked something) ends after a configurable time without speech."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from aioesphomeapi.api_pb2 import VoiceAssistantAudio  # type: ignore[attr-defined]

from linux_voice_assistant import satellite as satellite_module
from linux_voice_assistant.satellite import VoiceSatelliteProtocol


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _satellite(monkeypatch, follow_up_timeout=5):
    clock = Clock()
    monkeypatch.setattr(satellite_module.time, "monotonic", clock)
    s = VoiceSatelliteProtocol.__new__(VoiceSatelliteProtocol)
    s.state = SimpleNamespace(muted=False, follow_up_timeout=follow_up_timeout)  # type: ignore[assignment]
    s.send_messages = MagicMock()  # type: ignore[method-assign]
    s._is_streaming_audio = True
    s._vad_started = False
    s._stream_open = False
    s._stream_started_at = 0.0
    s._silence_limit = None
    s._followup_stream = False
    return s, clock


def _ends(s) -> int:
    return sum(1 for c in s.send_messages.call_args_list for m in c.args[0] if isinstance(m, VoiceAssistantAudio) and m.end)


def _stream(s, clock, seconds, step=0.064):
    for _ in range(int(round(seconds / step))):
        clock.now += step
        s.handle_audio(b"\x00" * 2048)


def test_follow_up_stream_ends_after_the_timeout(monkeypatch):
    s, clock = _satellite(monkeypatch)
    s._followup_stream = True
    _stream(s, clock, 8.0)
    assert _ends(s) == 1
    assert not s._is_streaming_audio
    assert not s._followup_stream  # one-shot: the next stream is a normal one


def test_ends_no_earlier_than_the_timeout(monkeypatch):
    s, clock = _satellite(monkeypatch, follow_up_timeout=5)
    s._followup_stream = True
    _stream(s, clock, 4.5)
    assert _ends(s) == 0


def test_normal_stream_has_no_timeout(monkeypatch):
    s, clock = _satellite(monkeypatch)
    _stream(s, clock, 12.0)
    assert _ends(s) == 0


def test_zero_leaves_it_to_home_assistant(monkeypatch):
    s, clock = _satellite(monkeypatch, follow_up_timeout=0)
    s._followup_stream = True
    _stream(s, clock, 12.0)
    assert _ends(s) == 0


def test_speech_detected_by_home_assistant_disables_the_timeout(monkeypatch):
    s, clock = _satellite(monkeypatch)
    s._followup_stream = True
    _stream(s, clock, 2.0)
    s._vad_started = True
    _stream(s, clock, 8.0)
    assert _ends(s) == 0
