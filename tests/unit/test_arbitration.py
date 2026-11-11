"""Tests for the arbitration between satellites that heard one wake word (arbitration.Arbiter)."""

import json
import time

from linux_voice_assistant.arbitration import Arbiter, parse_peers

PEER = ("10.0.0.2", 6056)


class Rig:
    """An arbiter with a hand-driven clock; the window's end runs when the test calls `end()`."""

    def __init__(self, name="room", offset_db=0.0, peers=(PEER,)):
        self.now = 100.0
        self.sent = []
        self.pending = []
        self.won = 0
        self.a = Arbiter(name, list(peers), 0.15, offset_db, clock=lambda: self.now, send=lambda d, p: self.sent.append((json.loads(d), p)),
                         schedule=lambda delay, fn: self.pending.append(fn))

    def peer(self, score, word="Alexa", name="mirror", address=PEER[0]):
        self.a.received(json.dumps({"name": name, "word": word, "score": score}).encode(), address)

    def claim(self, snr, word="Alexa"):
        self.a.claim(word, snr, self._win)

    def _win(self):
        self.won += 1

    def end(self):
        for fn in self.pending:
            fn()
        self.pending.clear()


def test_the_higher_score_starts_and_the_claim_goes_to_every_peer():
    r = Rig()
    r.claim(35)
    assert r.sent == [({"name": "room", "word": "Alexa", "score": 35.0}, PEER)]
    r.now += 0.05
    r.peer(30)
    r.end()
    assert r.won == 1


def test_a_peer_that_heard_it_better_takes_it_whether_it_claimed_before_or_after():
    # 2026-11-11 living room: the hallway reported first with SNR 30, the room 62 ms later with 35; the hallway must yield
    before = Rig(name="mirror")
    before.peer(35, name="room")
    before.now += 0.06
    before.claim(30)
    before.end()
    after = Rig(name="mirror")
    after.claim(30)
    after.now += 0.06
    after.peer(35, name="room")
    after.end()
    assert before.won == after.won == 0


def test_the_offset_from_the_configuration_counts():
    # 2026-11-11 bedroom: the hallway's microphone heard it with SNR 39, the bedroom's with 35; the hallway's offset -6 gives it to the bedroom
    r = Rig(name="mirror", offset_db=-6)
    r.claim(39)
    r.peer(35, name="bedroom")
    r.end()
    assert r.won == 0 and r.sent[0][0]["score"] == 33.0


def test_claims_outside_the_window_or_of_another_word_do_not_compete():
    r = Rig()
    r.peer(50)                                    # an earlier wake word
    r.now += 1.0
    r.peer(50, word="Hey Jarvis")
    r.claim(20)
    r.end()
    assert r.won == 1


def test_only_configured_peers_count():
    r = Rig()
    r.peer(50, address="10.0.0.99")
    r.a.received(b"not json", PEER[0])
    r.claim(20)
    r.end()
    assert r.won == 1


def test_equal_scores_go_to_the_same_satellite_on_both_sides():
    a, b = Rig(name="bedroom"), Rig(name="mirror")
    a.claim(30)
    a.peer(30, name="mirror")
    b.claim(30)
    b.peer(30, name="bedroom")
    a.end()
    b.end()
    assert (a.won, b.won) == (1, 0)


def test_without_peers_or_history_it_starts_at_once():
    lone = Rig(peers=())
    lone.claim(20)
    early = Rig()
    early.claim(None)                              # the satellite just started: no history for an SNR
    assert lone.won == early.won == 1 and not early.sent and not lone.pending


def test_two_arbiters_talk_over_udp():
    won = []
    a = Arbiter("room", [("127.0.0.1", 46102)], 0.2)
    b = Arbiter("mirror", [("127.0.0.1", 46101)], 0.2)
    a.start(46101)
    b.start(46102)
    a.claim("Alexa", 35, lambda: won.append("room"))
    b.claim("Alexa", 30, lambda: won.append("mirror"))
    deadline = time.monotonic() + 2
    while not won and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)
    assert won == ["room"]


def test_peers_are_parsed_with_the_default_port():
    assert parse_peers("127.0.0.1, 127.0.0.2:7000,", 6056) == [("127.0.0.1", 6056), ("127.0.0.2", 7000)]
