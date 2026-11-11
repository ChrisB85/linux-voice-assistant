"""Which satellite answers a wake word that several of them heard: the one that heard it best, not the first to report it.

Home Assistant keeps the first satellite to report a wake word and turns the others away (duplicate_wake_up_detected). In neighbouring
rooms that is close to random: on 2026-11-11 the hallway took 2 of 4 wake words said in the living room, reporting ~50 ms earlier.
Here each satellite that detects a wake word sends its score (wake_metrics.wake_snr plus its own offset) to its peers over UDP, waits
`window_s`, and starts the conversation only when no peer claimed the same word with a higher score within `window_s` before or after
its own claim. A satellite with no peers, or whose peers are down, starts after the window as before.
Only datagrams from the configured peers' addresses count.
"""

import json
import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

_MAX_CLAIMS = 64  # claims kept from peers; only those of the last window matter


@dataclass(frozen=True)
class Claim:
    name: str
    word: str
    score: float
    t: float  # local monotonic time the claim was made or received


def parse_peers(text: str, default_port: int) -> List[Tuple[str, int]]:
    """"host[:port],host[:port]" into (address, port); a host is resolved once, at start."""
    peers = []
    for item in (p.strip() for p in text.split(",")):
        if not item:
            continue
        host, _, port = item.partition(":")
        peers.append((socket.gethostbyname(host), int(port) if port else default_port))
    return peers


def beats(a: Claim, b: Claim) -> bool:
    """Whether claim `a` wins over `b`: the higher score; the same score goes to the name first in order, so every satellite agrees."""
    return (a.score, b.name) > (b.score, a.name)


class Arbiter:
    """Claims of one satellite and of its peers; `claim` calls `on_win` after the window unless a peer's claim beats it."""

    def __init__(
        self,
        name: str,
        peers: List[Tuple[str, int]],
        window_s: float,
        offset_db: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        send: Optional[Callable[[bytes, Tuple[str, int]], None]] = None,
        schedule: Optional[Callable[[float, Callable[[], None]], None]] = None,
    ) -> None:
        self.name = name
        self.peers = peers
        self.window_s = window_s
        self.offset_db = offset_db
        self._clock = clock
        self._send = send
        self._schedule = schedule or _timer
        self._claims: Deque[Claim] = deque(maxlen=_MAX_CLAIMS)
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None

    def start(self, port: int) -> None:
        """Listen for the peers' claims on UDP `port` (a daemon thread) and send own claims from the same socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        self._sock = sock
        if self._send is None:
            self._send = lambda data, addr: sock.sendto(data, addr)
        threading.Thread(target=self._receive_loop, daemon=True, name="arbitration").start()
        _LOGGER.info("Arbitration listening on UDP %d, peers %s, window %.0f ms, offset %+.1f dB", port, self.peers, self.window_s * 1000, self.offset_db)

    def _receive_loop(self) -> None:
        assert self._sock is not None
        while True:
            try:
                data, addr = self._sock.recvfrom(1024)
            except OSError:
                return
            self.received(data, addr[0])

    def received(self, data: bytes, address: str) -> None:
        """A peer's claim; datagrams from other addresses or not in the claim's shape are dropped."""
        if address not in {a for a, _ in self.peers}:
            return
        try:
            msg = json.loads(data)
            claim = Claim(str(msg["name"]), str(msg["word"]), float(msg["score"]), self._clock())
        except (ValueError, KeyError, TypeError):
            return
        with self._lock:
            self._claims.append(claim)

    def claim(self, word: str, snr: Optional[float], on_win: Callable[[], None]) -> None:
        """Tell the peers this satellite heard `word` with this SNR and decide after the window. No SNR (too little history) or no peers:
        `on_win` at once, as without arbitration."""
        if snr is None or not self.peers:
            on_win()
            return
        own = Claim(self.name, word, snr + self.offset_db, self._clock())
        packet = json.dumps({"name": own.name, "word": own.word, "score": own.score}).encode()
        for peer in self.peers:
            try:
                if self._send is not None:
                    self._send(packet, peer)
            except OSError as err:
                _LOGGER.debug("Arbitration: could not reach %s: %s", peer, err)
        self._schedule(self.window_s, lambda: self._decide(own, on_win))

    def _decide(self, own: Claim, on_win: Callable[[], None]) -> None:
        with self._lock:
            rivals = [c for c in self._claims if c.word == own.word and abs(c.t - own.t) <= self.window_s]
        winner = next((c for c in sorted(rivals, key=lambda c: (-c.score, c.name)) if beats(c, own)), None)
        if winner is None:
            _LOGGER.info("Arbitration: won %s score=%.1f against %s", own.word, own.score, [(c.name, c.score) for c in rivals])
            on_win()
        else:
            _LOGGER.info("Arbitration: yielded %s score=%.1f to %s score=%.1f", own.word, own.score, winner.name, winner.score)


def _timer(delay: float, fn: Callable[[], None]) -> None:
    t = threading.Timer(delay, fn)
    t.daemon = True
    t.start()
