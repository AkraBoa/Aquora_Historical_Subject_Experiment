"""Minimal historical-disposition experiment.

No persona. No LLM. No semantic retrieval.
The point is to test whether an append-only interaction history can become
causally relevant to later behavior through state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import exp
from typing import Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class Event:
    step: int
    context: str
    action: str
    outcome: float  # observed consequence in [-1, +1]
    note: str = ""
    previous_digest: str = "GENESIS"

    @property
    def digest(self) -> str:
        raw = f"{self.step}|{self.context}|{self.action}|{self.outcome:.12f}|{self.note}|{self.previous_digest}"
        return sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    step: int
    weights: Tuple[Tuple[str, float], ...]
    event_digest: str


@dataclass
class HistoricalSubject:
    learning_rate: float = 0.35
    decay: float = 0.985
    _weights: Dict[str, float] = field(default_factory=dict)
    _events: List[Event] = field(default_factory=list)
    _trajectory: List[Snapshot] = field(default_factory=list)

    @property
    def events(self) -> Tuple[Event, ...]:
        return tuple(self._events)

    @property
    def trajectory(self) -> Tuple[Snapshot, ...]:
        return tuple(self._trajectory)

    @property
    def disposition(self) -> Mapping[str, float]:
        return dict(self._weights)

    def experience(self, context: str, action: str, outcome: float, note: str = "") -> Event:
        if not -1.0 <= outcome <= 1.0:
            raise ValueError("outcome must be in [-1, 1]")

        # History changes the current system. We do not rewrite a prior event.
        for key in tuple(self._weights):
            self._weights[key] *= self.decay

        key = self._key(context, action)
        old = self._weights.get(key, 0.0)
        self._weights[key] = old + self.learning_rate * (outcome - old)

        prev = self._events[-1].digest if self._events else "GENESIS"
        event = Event(
            step=len(self._events) + 1,
            context=context,
            action=action,
            outcome=outcome,
            note=note,
            previous_digest=prev,
        )
        self._events.append(event)
        self._trajectory.append(
            Snapshot(
                step=event.step,
                weights=tuple(sorted(self._weights.items())),
                event_digest=event.digest,
            )
        )
        return event

    def preference(self, context: str, action: str) -> float:
        return self._weights.get(self._key(context, action), 0.0)

    def choose(self, context: str, actions: List[str]) -> str:
        if not actions:
            raise ValueError("at least one action is required")
        # Deterministic on purpose: differences should come from history, not RNG.
        scored = [(self.preference(context, action), action) for action in actions]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored[0][1]

    def confidence(self, context: str, action: str) -> float:
        """Maps absolute learned preference to [0.5, 1). Not epistemic certainty."""
        x = abs(self.preference(context, action))
        return 1.0 / (1.0 + exp(-x))

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for event in self._events:
            if event.previous_digest != prev:
                return False
            prev = event.digest
        return True

    @staticmethod
    def _key(context: str, action: str) -> str:
        return f"{context}\x1f{action}"
