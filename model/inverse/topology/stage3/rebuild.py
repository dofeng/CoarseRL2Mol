"""Structured failures at the Layer1/2 candidate-selection boundary."""

from copy import deepcopy
from typing import Any, Dict, Sequence


class Layer12CandidatesExhausted(RuntimeError):
    """No candidate survived; retain every rejection for the caller."""

    def __init__(self, stage: str, candidates: Sequence[Dict[str, Any]]):
        self.proof = {
            "ok": False,
            "reason": "layer12_candidates_exhausted",
            "failure_domain": "candidate_selection",
            "stage": str(stage),
            "candidates": deepcopy(list(candidates)),
        }
        super().__init__(f"layer12_candidates_exhausted:{stage}")


def rebuild_exception_proof(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, Layer12CandidatesExhausted):
        return deepcopy(exc.proof)
    return {
        "ok": False,
        "reason": "layer12_rebuild_exception",
        "failure_domain": "execution",
        "exception_type": type(exc).__name__,
        "error": str(exc),
    }
