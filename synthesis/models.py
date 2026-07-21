"""Small typed data models shared by the synthesis modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Strategy(StrEnum):
    ELIMINATION = "elimination"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True, slots=True)
class RandomizedExample:
    source_index: int
    question: str
    answer: str
    identifier_mapping: dict[str, str]
    variant_index: int = 0


@dataclass(frozen=True, slots=True)
class Trajectory:
    strategy: Strategy
    text: str
    prediction: str
