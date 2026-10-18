"""A cut-off TTS response must leave a trace in the journal at the default INFO level."""

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from linux_voice_assistant.player.libmpv import LibMpvPlayer
from linux_voice_assistant.player.state import PlayerState
from linux_voice_assistant.satellite import VoiceSatelliteProtocol


def _satellite() -> VoiceSatelliteProtocol:
    satellite = VoiceSatelliteProtocol.__new__(VoiceSatelliteProtocol)
    satellite.state = SimpleNamespace(  # type: ignore[assignment]
        tts_player=MagicMock(),
        active_wake_words={"stop"},
        stop_word=SimpleNamespace(id="stop"),
    )
    satellite._timer_finished = False
    return satellite


def _player() -> LibMpvPlayer:
    player = LibMpvPlayer.__new__(LibMpvPlayer)
    player._log = logging.getLogger("LibMpvPlayer")
    player._state = PlayerState.PLAYING
    player._state_lock = threading.Lock()
    player._done_callback = None
    return player


def test_stopping_tts_logs_at_info(caplog):
    satellite = _satellite()
    with caplog.at_level(logging.INFO):
        satellite.stop()

    assert any(r.levelno == logging.INFO and "TTS response stopped manually" in r.getMessage() for r in caplog.records)


def test_end_of_file_logs_its_reason_at_info(caplog):
    player = _player()
    with caplog.at_level(logging.INFO):
        player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=2)))

    assert any(r.levelno == logging.INFO and "reason=2" in r.getMessage() for r in caplog.records)
