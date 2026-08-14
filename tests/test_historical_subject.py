from src.historical_subject import HistoricalSubject


def test_identical_origins_diverge_through_history():
    a = HistoricalSubject()
    b = HistoricalSubject()

    # Same initial system, different lived consequences.
    for _ in range(4):
        a.experience("stranger_approaches", "engage", +0.9)
        a.experience("stranger_approaches", "withdraw", -0.4)

        b.experience("stranger_approaches", "engage", -0.8)
        b.experience("stranger_approaches", "withdraw", +0.7)

    same_context_actions = ["engage", "withdraw"]
    assert a.choose("stranger_approaches", same_context_actions) == "engage"
    assert b.choose("stranger_approaches", same_context_actions) == "withdraw"


def test_revision_changes_present_without_erasing_past():
    s = HistoricalSubject(learning_rate=0.6)

    for _ in range(3):
        s.experience("door", "open", -1.0, "opening had a bad consequence")
    old_events = s.events
    assert s.choose("door", ["open", "wait"]) == "wait"

    # New experience can revise the disposition; old history remains.
    for _ in range(8):
        s.experience("door", "open", +1.0, "context changed; opening now helps")

    assert s.choose("door", ["open", "wait"]) == "open"
    assert s.events[: len(old_events)] == old_events
    assert len(s.events) > len(old_events)
    assert s.verify_chain()


def test_order_becomes_part_of_history():
    a = HistoricalSubject(learning_rate=0.7, decay=0.9)
    b = HistoricalSubject(learning_rate=0.7, decay=0.9)

    # Same multiset of outcomes, opposite temporal order.
    for outcome in [-1.0, -1.0, +1.0, +1.0]:
        a.experience("signal", "follow", outcome)
    for outcome in [+1.0, +1.0, -1.0, -1.0]:
        b.experience("signal", "follow", outcome)

    assert a.preference("signal", "follow") > 0
    assert b.preference("signal", "follow") < 0


def test_no_persona_is_required():
    s = HistoricalSubject()
    assert s.disposition == {}
    assert s.events == ()
    assert s.choose("unknown", ["a", "b"]) == "a"  # neutral deterministic tie-break


def test_event_chain_is_append_only_traceable():
    s = HistoricalSubject()
    s.experience("x", "a", 0.5)
    s.experience("x", "b", -0.5)
    assert s.verify_chain()
    assert s.events[1].previous_digest == s.events[0].digest
