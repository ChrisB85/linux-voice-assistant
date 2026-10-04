"""End-of-speech for short words that Home Assistant's VAD never notices.

HA's voice-command segmenter needs 0.3 s of cumulative speech before it starts a command, so a single short word
("dwa", "tak", "nie") never starts it and the stream only ends at the 15 s timeout. A recording of such a word shows a
clear burst (about 20x above the background) followed by silence. When HA's VAD has not started, this detector spots
that burst and, after a second of quiet, tells the caller to send ``VoiceAssistantAudio(end=True)`` so HA stops the
audio stream and runs STT on what it has. Once HA's VAD has started it stays out of the way.
"""

import statistics
from typing import List, Optional

import numpy as np

RATE = 16000
WIDTH = 2  # bytes per sample


class ShortBurstEndpointer:
    def __init__(
        self,
        min_peak: float = 250.0,    # absolute RMS (64 ms blocks) a burst must reach; speech peaks ~700-2700, room noise ~30-260
        factor: float = 6.0,        # and it must stand out this much from the stream's median level
        quiet_s: float = 1.0,       # silence after the burst before the end is requested
        history_s: float = 20.0,
        ignore_s: float = 1.0,      # the chime / echo of the question right after the mic opens is not a spoken word
        no_speech_s: Optional[float] = None,  # end a stream with no speech at all after this long (None: leave it to HA's 15 s)
    ) -> None:
        self._min_peak = min_peak
        self._factor = factor
        self._quiet_s = quiet_s
        self._ignore_s = ignore_s
        self.no_speech_s = no_speech_s
        self._max_blocks = int(history_s * RATE / 1024) + 1
        self.reset()

    def reset(self) -> None:
        self._levels: List[float] = []
        self._elapsed = 0.0
        self.burst_at = 0.0  # stream time (s) when the burst that ended the stream began, for logs
        self._burst = False
        self._quiet = 0.0
        self._requested = False
        self.silent_end = False  # the end was requested because nothing was said at all (no_speech_s)

    def feed(self, chunk: bytes, ha_vad_started: bool) -> bool:
        """Feed one microphone block; True exactly once per stream when the end of audio should be sent."""
        if ha_vad_started or self._requested or not chunk:
            return False
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float64)
        level = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        self._levels.append(level)
        if len(self._levels) > self._max_blocks:
            self._levels.pop(0)
        seconds = samples.size / RATE
        threshold = max(self._min_peak, self._factor * statistics.median(self._levels))
        started = self._elapsed
        self._elapsed += seconds
        if started < self._ignore_s:
            return False
        if not self._burst and self.no_speech_s is not None and self._elapsed >= self.no_speech_s and level < threshold:
            self._requested = True
            self.silent_end = True
            return True
        if level >= threshold:
            if not self._burst:
                self.burst_at = started
            self._burst = True
            self._quiet = 0.0
        elif self._burst:
            self._quiet += seconds
            if self._quiet >= self._quiet_s:
                self._requested = True
                return True
        return False
