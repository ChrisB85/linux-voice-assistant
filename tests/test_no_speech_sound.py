"""Sound feedback when the pipeline ends because no speech was recognized."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from aioesphomeapi.model import VoiceAssistantEventType

from linux_voice_assistant.satellite import VoiceSatelliteProtocol

NO_TEXT = {"code": "stt-no-text-recognized", "message": "No text recognized"}
DUPLICATE = {"code": "duplicate_wake_up_detected", "message": "Duplicate wake-up detected for Alexa"}


def _satellite(no_speech_sound: str = "no_speech.flac") -> VoiceSatelliteProtocol:
    satellite = VoiceSatelliteProtocol.__new__(VoiceSatelliteProtocol)
    satellite.state = SimpleNamespace(  # type: ignore[assignment]
        no_speech_sound=no_speech_sound,
        no_speech_sound_codes={"stt-no-text-recognized"},
        thinking_sound_enabled=False,
        tts_player=MagicMock(),
        music_player=MagicMock(),
        active_wake_words=set(),
        stop_word=SimpleNamespace(id="stop"),
    )
    satellite.send_messages = MagicMock()  # type: ignore[method-assign]
    satellite._emit = MagicMock()  # type: ignore[method-assign]
    satellite._timer_finished = False
    return satellite


def _run(satellite: VoiceSatelliteProtocol, error: dict) -> None:
    satellite.handle_voice_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, {})
    satellite.handle_voice_event(VoiceAssistantEventType.VOICE_ASSISTANT_ERROR, error)
    satellite.handle_voice_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})


def test_plays_sound_then_goes_idle():
    satellite = _satellite()
    _run(satellite, NO_TEXT)

    play_call = satellite.state.tts_player.play.call_args
    assert play_call.args[0] == "no_speech.flac"
    # Mic stays blocked until the sound ends, so the speaker cannot wake it.
    assert satellite._pipeline_active

    play_call.kwargs["done_callback"]()
    assert not satellite._pipeline_active
    satellite.send_messages.assert_called_once()


def test_other_error_codes_stay_silent():
    # The satellite that lost the wake word race must not beep.
    satellite = _satellite()
    _run(satellite, DUPLICATE)

    satellite.state.tts_player.play.assert_not_called()
    assert not satellite._pipeline_active


def test_empty_sound_disables_feature():
    satellite = _satellite(no_speech_sound="")
    _run(satellite, NO_TEXT)

    satellite.state.tts_player.play.assert_not_called()


def test_next_run_without_error_stays_silent():
    satellite = _satellite()
    _run(satellite, NO_TEXT)
    satellite.state.tts_player.play.reset_mock()

    satellite.handle_voice_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, {})
    satellite.handle_voice_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})
    satellite.state.tts_player.play.assert_not_called()


def test_ringing_timer_wins_over_sound(monkeypatch):
    satellite = _satellite()
    satellite._timer_finished = True
    resumed = MagicMock()
    monkeypatch.setattr(satellite, "_play_timer_finished", resumed)
    _run(satellite, NO_TEXT)

    resumed.assert_called_once()
    satellite.state.tts_player.play.assert_not_called()
