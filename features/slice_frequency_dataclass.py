from dataclasses import dataclass


@dataclass(frozen=True)
class SliceFrequency:
    start: int
    stop: int
