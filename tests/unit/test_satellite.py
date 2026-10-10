"""Unit tests for VoiceSatelliteProtocol logic."""

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.conftest import make_satellite, make_state

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_satellite_stored_on_state(self, tmp_path):
        sat = make_satellite(tmp_path)
        assert sat.state.satellite is sat

    def test_connected_starts_false(self, tmp_path):
        sat = make_satellite(tmp_path)
        assert sat.state.connected is False

    def test_media_player_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MediaPlayerEntity

        sat = make_satellite(tmp_path)
        assert sat.state.media_player_entity is not None
        assert isinstance(sat.state.media_player_entity, MediaPlayerEntity)

    def test_mute_switch_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MuteSwitchEntity

        sat = make_satellite(tmp_path)
        assert sat.state.mute_switch_entity is not None
        assert isinstance(sat.state.mute_switch_entity, MuteSwitchEntity)

    def test_thinking_sound_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import ThinkingSoundEntity

        sat = make_satellite(tmp_path)
        assert sat.state.thinking_sound_entity is not None
        assert isinstance(sat.state.thinking_sound_entity, ThinkingSoundEntity)

    def test_follow_up_sound_switch_created_disabled(self, tmp_path):
        from linux_voice_assistant.entity import ContinueConversationSoundEntity

        sat = make_satellite(tmp_path)
        assert isinstance(sat.state.continue_conversation_sound_entity, ContinueConversationSoundEntity)
        assert sat.state.continue_conversation_sound_enabled is False

    def test_mic_gain_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MicSettingEntity

        sat = make_satellite(tmp_path)
        assert sat.state.mic_gain_entity is not None
        assert isinstance(sat.state.mic_gain_entity, MicSettingEntity)

    def test_mic_noise_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MicSettingEntity

        sat = make_satellite(tmp_path)
        assert sat.state.mic_noise_suppression_entity is not None
        assert isinstance(sat.state.mic_noise_suppression_entity, MicSettingEntity)

    def test_mic_volume_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MicSettingEntity

        sat = make_satellite(tmp_path)
        assert sat.state.mic_volume_entity is not None
        assert isinstance(sat.state.mic_volume_entity, MicSettingEntity)

    def test_follow_up_timeout_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MicSettingEntity

        sat = make_satellite(tmp_path)
        entity = sat.state.follow_up_timeout_entity
        assert isinstance(entity, MicSettingEntity)
        assert entity.object_id == "follow_up_timeout"
        assert entity in sat.state.entities

    def test_follow_up_timeout_set_from_home_assistant_is_saved(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.follow_up_timeout_entity._set_value(8.0)
        assert sat.state.follow_up_timeout == 8
        assert sat.state.preferences.follow_up_timeout == 8
        assert '"follow_up_timeout": 8' in (tmp_path / "preferences.json").read_text()

    def test_follow_up_timeout_is_clamped(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.persist_follow_up_timeout(99)
        assert sat.state.follow_up_timeout == 15
        sat.state.persist_follow_up_timeout(-3)
        assert sat.state.follow_up_timeout == 0

    def test_listen_timeout_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import MicSettingEntity

        sat = make_satellite(tmp_path)
        entity = sat.state.listen_timeout_entity
        assert isinstance(entity, MicSettingEntity)
        assert entity.object_id == "listen_timeout"
        assert entity in sat.state.entities
        assert sat.state.listen_timeout == 0  # default leaves it to Home Assistant

    def test_listen_timeout_set_from_home_assistant_is_saved(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.listen_timeout_entity._set_value(3.0)
        assert sat.state.listen_timeout == 3
        assert '"listen_timeout": 3' in (tmp_path / "preferences.json").read_text()
        sat.state.persist_listen_timeout(99)
        assert sat.state.listen_timeout == 15

    def test_pipeline_not_active_on_start(self, tmp_path):
        sat = make_satellite(tmp_path)
        assert sat._pipeline_active is False

    def test_not_streaming_audio_on_start(self, tmp_path):
        sat = make_satellite(tmp_path)
        assert sat._is_streaming_audio is False

    def test_not_muted_on_start(self, tmp_path):
        sat = make_satellite(tmp_path)
        assert sat.state.muted is False

    def test_thinking_sound_loaded_from_preferences(self, tmp_path):
        state = make_state(tmp_path)
        state.preferences.thinking_sound = 1

        with (
            patch("linux_voice_assistant.satellite.WakeWord1SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.WakeWord2SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.StopWordSensitivityNumberEntity", MagicMock()),
        ):
            from linux_voice_assistant.satellite import VoiceSatelliteProtocol

            sat = VoiceSatelliteProtocol(state)

        assert sat.state.thinking_sound_enabled is True

    def test_follow_up_sound_loaded_from_preferences(self, tmp_path):
        state = make_state(tmp_path)
        state.preferences.continue_conversation_sound = 1

        with (
            patch("linux_voice_assistant.satellite.WakeWord1SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.WakeWord2SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.StopWordSensitivityNumberEntity", MagicMock()),
        ):
            from linux_voice_assistant.satellite import VoiceSatelliteProtocol

            sat = VoiceSatelliteProtocol(state)

        assert sat.state.continue_conversation_sound_enabled is True

    def test_output_only_sets_limited_features(self, tmp_path):
        from aioesphomeapi.model import VoiceAssistantFeature

        state = make_state(tmp_path)
        state.output_only = True

        with (
            patch("linux_voice_assistant.satellite.WakeWord1SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.WakeWord2SensitivityNumberEntity", MagicMock()),
            patch("linux_voice_assistant.satellite.StopWordSensitivityNumberEntity", MagicMock()),
        ):
            from linux_voice_assistant.satellite import VoiceSatelliteProtocol

            sat = VoiceSatelliteProtocol(state)

        assert sat.supported_features & VoiceAssistantFeature.VOICE_ASSISTANT == 0


# ---------------------------------------------------------------------------
# _set_muted()
# ---------------------------------------------------------------------------


class TestSetMuted:
    def test_muting_sets_muted_flag(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_muted(True)
        assert sat.state.muted is True

    def test_unmuting_clears_muted_flag(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.muted = True
        sat._set_muted(False)
        assert sat.state.muted is False

    def test_muting_stops_tts_player(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_muted(True)
        sat.state.tts_player.stop.assert_called()

    def test_muting_plays_mute_sound(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_muted(True)
        sat.state.tts_player.play.assert_called_with(sat.state.mute_sound)

    def test_unmuting_plays_unmute_sound(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_muted(False)
        sat.state.tts_player.play.assert_called_with(sat.state.unmute_sound)

    def test_muting_stops_audio_streaming(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._is_streaming_audio = True
        sat._set_muted(True)
        assert sat._is_streaming_audio is False


# ---------------------------------------------------------------------------
# _set_thinking_sound_enabled()
# ---------------------------------------------------------------------------


class TestSetThinkingSoundEnabled:
    def test_enables_thinking_sound(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_thinking_sound_enabled(True)
        assert sat.state.thinking_sound_enabled is True

    def test_disables_thinking_sound(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.thinking_sound_enabled = True
        sat._set_thinking_sound_enabled(False)
        assert sat.state.thinking_sound_enabled is False

    def test_updates_preferences(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_thinking_sound_enabled(True)
        assert sat.state.preferences.thinking_sound == 1

    def test_saves_preferences_to_file(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_thinking_sound_enabled(True)
        assert sat.state.preferences_path.exists()


# ---------------------------------------------------------------------------
# _set_sensitivity_1/2 and _set_stop_sensitivity
# ---------------------------------------------------------------------------


class TestSensitivitySetters:
    def test_set_sensitivity_1_updates_threshold(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_sensitivity_1(0.85)
        assert sat.state.wake_word_1_threshold == pytest.approx(0.85)

    def test_set_sensitivity_1_updates_preferences(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_sensitivity_1(0.85)
        assert sat.state.preferences.wake_word_1_sensitivity == pytest.approx(0.85)

    def test_set_sensitivity_2_updates_threshold(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_sensitivity_2(0.6)
        assert sat.state.wake_word_2_threshold == pytest.approx(0.6)

    def test_set_sensitivity_2_updates_preferences(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_sensitivity_2(0.6)
        assert sat.state.preferences.wake_word_2_sensitivity == pytest.approx(0.6)

    def test_set_stop_sensitivity_updates_threshold(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_stop_sensitivity(0.5)
        assert sat.state.stop_word_threshold == pytest.approx(0.5)

    def test_set_stop_sensitivity_updates_preferences(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_stop_sensitivity(0.5)
        assert sat.state.preferences.stop_word_sensitivity == pytest.approx(0.5)

    def test_sensitivity_saves_preferences(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._set_sensitivity_1(0.9)
        assert sat.state.preferences_path.exists()


# ---------------------------------------------------------------------------
# handle_audio()
# ---------------------------------------------------------------------------


class TestHandleAudio:
    def test_does_not_send_when_not_streaming(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._is_streaming_audio = False
        sat.handle_audio(b"\x00" * 320)
        sat._writelines.assert_not_called()

    def test_does_not_send_when_muted(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._is_streaming_audio = True
        sat.state.muted = True
        sat.handle_audio(b"\x00" * 320)
        sat._writelines.assert_not_called()

    def test_sends_when_streaming_and_not_muted(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._is_streaming_audio = True
        sat.state.muted = False
        sat._loop = None
        sat.handle_audio(b"\x00" * 320)
        sat._writelines.assert_called()


# ---------------------------------------------------------------------------
# play_tts()
# ---------------------------------------------------------------------------


class TestPlayTts:
    def test_does_not_play_when_no_url(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._tts_url = None
        sat.play_tts()
        sat.state.tts_player.play.assert_not_called()

    def test_does_not_play_when_already_played(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._tts_url = "http://example.com/tts.mp3"
        sat._tts_played = True
        sat.play_tts()
        sat.state.tts_player.play.assert_not_called()

    def test_plays_tts_url(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._tts_url = "http://example.com/tts.mp3"
        sat._tts_played = False
        sat.play_tts()
        sat.state.tts_player.play.assert_called_once()
        args, _ = sat.state.tts_player.play.call_args
        assert args[0] == "http://example.com/tts.mp3"

    def test_sets_tts_played_flag(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._tts_url = "http://example.com/tts.mp3"
        sat._tts_played = False
        sat.play_tts()
        assert sat._tts_played is True

    def test_adds_stop_word_to_active_wake_words(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._tts_url = "http://example.com/tts.mp3"
        sat._tts_played = False
        sat.play_tts()
        assert sat.state.stop_word.id in sat.state.active_wake_words


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_clears_pipeline_active(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._pipeline_active = True
        sat.stop()
        assert sat._pipeline_active is False

    def test_stop_discards_stop_word_from_active(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.active_wake_words.add(sat.state.stop_word.id)
        sat.stop()
        assert sat.state.stop_word.id not in sat.state.active_wake_words

    def test_stop_calls_tts_player_stop(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._timer_finished = False
        sat.stop()
        sat.state.tts_player.stop.assert_called()


# ---------------------------------------------------------------------------
# duck() / unduck()
# ---------------------------------------------------------------------------


class TestDuckUnduck:
    def test_duck_calls_music_player_duck(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.duck()
        sat.state.music_player.duck.assert_called_once()

    def test_unduck_calls_music_player_unduck(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.unduck()
        sat.state.music_player.unduck.assert_called_once()


# ---------------------------------------------------------------------------
# connection_lost()
# ---------------------------------------------------------------------------


class TestConnectionLost:
    def test_connection_lost_clears_connected_flag(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.connected = True
        sat.connection_lost(None)
        assert sat.state.connected is False

    def test_connection_lost_clears_satellite_reference(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.connection_lost(None)
        assert sat.state.satellite is None

    def test_connection_lost_stops_streaming(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._is_streaming_audio = True
        sat.connection_lost(None)
        assert sat._is_streaming_audio is False

    def test_connection_lost_clears_pipeline_active(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat._pipeline_active = True
        sat.connection_lost(None)
        assert sat._pipeline_active is False

    def test_connection_lost_stops_music_player(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.connection_lost(None)
        sat.state.music_player.stop.assert_called()

    def test_connection_lost_stops_tts_player(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.connection_lost(None)
        sat.state.tts_player.stop.assert_called()


# ---------------------------------------------------------------------------
# Named timers
# ---------------------------------------------------------------------------


def _finish_timer(sat, name):
    from aioesphomeapi.model import VoiceAssistantTimerEventType

    msg = MagicMock(timer_id="t1", total_seconds=600, seconds_left=0)
    msg.name = name
    sat.send_messages = MagicMock()
    sat.state.tts_player.is_playing = False
    sat.handle_timer_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED, msg)


def _sent_event(sat):
    (sent,), _ = sat.send_messages.call_args
    assert sent[0].service == "esphome.lva_timer_finished"
    assert sent[0].is_event is True
    return {d.key: d.value for d in sent[0].data}


class TestNamedTimer:
    def test_no_event_while_ringing(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat._play_timer_finished()
        sat.send_messages.assert_not_called()

    def test_event_after_ring_limit(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat._timer_ring_start -= sat.state.timer_max_ring_seconds
        sat._play_timer_finished()
        assert _sent_event(sat) == {"name": "piwo"}

    def test_event_after_stop_word(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat.stop()
        assert _sent_event(sat) == {"name": "piwo"}

    def test_event_fires_once(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat.stop()
        sat.send_messages.reset_mock()
        sat._play_timer_finished()
        sat.send_messages.assert_not_called()

    def test_unnamed_timer_fires_no_event(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "")
        sat.stop()
        sat.send_messages.assert_not_called()

    def test_announcement_during_ring_resumes_ring(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat.state.tts_player.play.reset_mock()
        sat._tts_finished()
        sat.state.tts_player.play.assert_called_once()
        assert sat.state.stop_word.id in sat.state.active_wake_words

    def test_ring_does_not_cut_announcement(self, tmp_path):
        sat = make_satellite(tmp_path)
        _finish_timer(sat, "piwo")
        sat.state.tts_player.play.reset_mock()
        sat.state.tts_player.is_playing = True
        sat._play_timer_finished()
        sat.state.tts_player.play.assert_not_called()


class TestFollowUpWakeWord:
    def _announce(self, sat, start_conversation=True):
        from aioesphomeapi.api_pb2 import VoiceAssistantAnnounceRequest  # type: ignore[attr-defined]

        sat.state.media_player_entity = MagicMock()
        msg = VoiceAssistantAnnounceRequest(media_id="http://x/q.mp3", text="q", start_conversation=start_conversation)
        list(sat.handle_message(msg))

    def test_entity_created(self, tmp_path):
        from linux_voice_assistant.entity import FollowUpWakeWordEntity

        sat = make_satellite(tmp_path)
        assert isinstance(sat.state.follow_up_wake_word_entity, FollowUpWakeWordEntity)
        assert sat.state.follow_up_wake_word_entity in sat.state.entities

    def test_text_command_from_home_assistant_reaches_the_entity(self, tmp_path):
        from aioesphomeapi.api_pb2 import TextCommandRequest, TextStateResponse  # type: ignore[attr-defined]

        sat = make_satellite(tmp_path)
        entity = sat.state.follow_up_wake_word_entity
        responses = list(sat.handle_message(TextCommandRequest(key=entity.key, state="Alexa")))
        assert entity.value == "Alexa"
        assert any(isinstance(r, TextStateResponse) and r.state == "Alexa" for r in responses)

    def test_start_conversation_uses_the_phrase_once(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.follow_up_wake_word_entity.value = "Alexa"
        self._announce(sat)
        assert sat._wake_word_phrase == "Alexa"
        assert sat.state.follow_up_wake_word_entity.value == ""
        self._announce(sat)
        assert sat._wake_word_phrase == ""

    def test_plain_announcement_leaves_the_phrase_alone(self, tmp_path):
        sat = make_satellite(tmp_path)
        sat.state.follow_up_wake_word_entity.value = "Alexa"
        self._announce(sat, start_conversation=False)
        assert sat._wake_word_phrase == ""
        assert sat.state.follow_up_wake_word_entity.value == "Alexa"
