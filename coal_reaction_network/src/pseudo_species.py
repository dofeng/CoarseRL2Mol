"""Minimal three-dimensional pseudo-species classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class FragmentProperties:
    """Properties of one connected molecular fragment."""

    carbon_count: int
    ring_count: int
    oxygen_count: int
    radical_sites: int = 0
    molecular_mass: float = 0.0

    def __post_init__(self) -> None:
        for name in ("carbon_count", "ring_count", "oxygen_count", "radical_sites"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.molecular_mass) or self.molecular_mass < 0:
            raise ValueError("molecular_mass must be non-negative")


@dataclass(frozen=True)
class PseudoSpecies:
    """A category node plus the current fragment's internal properties."""

    node_id: str
    carbon_bin: str
    ring_bin: str
    oxygen_bin: str
    fragment: FragmentProperties

    @property
    def descriptor(self) -> str:
        return f"C={self.carbon_bin};R={self.ring_bin};O={self.oxygen_bin}"


class ThreeDimensionalClassifier:
    """Classify using only carbon count, ring count, and oxygen count."""

    def __init__(
        self,
        carbon_bins: tuple[tuple[str, int, int | None], ...] | None = None,
        ring_bins: tuple[tuple[str, int, int | None], ...] | None = None,
        oxygen_bins: tuple[tuple[str, int, int | None], ...] | None = None,
    ) -> None:
        self.carbon_bins = carbon_bins or (
            ("C0", 0, 0), ("C1-5", 1, 5), ("C6-20", 6, 20),
            ("C21-40", 21, 40), ("C40+", 41, None),
        )
        self.ring_bins = ring_bins or (
            ("R0", 0, 0), ("R1", 1, 1), ("R2-5", 2, 5),
            ("R6-10", 6, 10), ("R10+", 11, None),
        )
        self.oxygen_bins = oxygen_bins or (
            ("O0", 0, 0), ("O1", 1, 1), ("O2", 2, 2),
            ("O3+", 3, None),
        )
        for bins, name in ((self.carbon_bins, "carbon"), (self.ring_bins, "ring"),
                           (self.oxygen_bins, "oxygen")):
            self._validate_bins(bins, name)
        self._node_ids: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def _validate_bins(bins: tuple[tuple[str, int, int | None], ...], name: str) -> None:
        if not bins:
            raise ValueError(f"{name}_bins cannot be empty")
        previous_max: int | None = None
        labels = set()
        for index, (label, lower, upper) in enumerate(bins):
            if not isinstance(lower, int) or isinstance(lower, bool) or (upper is not None and (not isinstance(upper, int) or isinstance(upper, bool))):
                raise ValueError(f"{name} bin boundaries must be integers")
            if not isinstance(label, str) or not label or "_" in label:
                raise ValueError(f"{name} labels must be nonempty strings without underscores")
            if label in labels or lower < 0 or (index == 0 and lower != 0):
                raise ValueError(f"invalid {name} bin: {label}")
            labels.add(label)
            if upper is None and index != len(bins) - 1:
                raise ValueError(f"Only the final {name} bin can be unbounded")
            if upper is not None and upper < lower:
                raise ValueError(f"invalid {name} bin: {label}")
            if previous_max is not None and lower != previous_max + 1:
                raise ValueError(f"{name}_bins must be contiguous and ordered")
            previous_max = upper
        if bins[-1][2] is not None:
            raise ValueError(f"Final {name} bin must cover all larger values")

    @staticmethod
    def _find_bin(value: int, bins: tuple[tuple[str, int, int | None], ...]) -> str:
        for label, lower, upper in bins:
            if value >= lower and (upper is None or value <= upper):
                return label
        raise ValueError(f"value {value} is outside the configured bins")

    def classify(self, fragment: FragmentProperties) -> PseudoSpecies:
        key = (self._find_bin(fragment.carbon_count, self.carbon_bins),
               self._find_bin(fragment.ring_count, self.ring_bins),
               self._find_bin(fragment.oxygen_count, self.oxygen_bins))
        if key not in self._node_ids:
            self._node_ids[key] = self._make_node_id(key)
        return PseudoSpecies(self._node_ids[key], *key, fragment)

    def node_definitions(self) -> dict[str, dict[str, str]]:
        return {node_id: {"carbon_bin": key[0], "ring_bin": key[1],
                          "oxygen_bin": key[2]}
                for key, node_id in self._node_ids.items()}

    @staticmethod
    def _make_node_id(key: tuple[str, str, str]) -> str:
        """Return a deterministic identifier shared by every frame and run."""
        return "_".join(key)

    def classify_mapping(self, values: Mapping[str, Any]) -> PseudoSpecies:
        return self.classify(FragmentProperties(**values))
