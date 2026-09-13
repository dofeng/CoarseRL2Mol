"""Inverse-driven hexagonal topology assembly public API."""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

from .input import initialize, initialize_from_inverse
from .models import AssemblyConfig, MCTSConfig, HeteroAromaticRequest
from .substitution import run_substitution_stage

# Local notebooks and older tests imported implementation modules directly.
# Keep those imports working while the public API moves to the ten-file layout.
from . import flex as _flex
from . import geometry as _geometry
from . import input as _input
from . import models as _models
from . import output as _output
from . import rigid as _rigid
from . import state as _state


def _load_runner():
    return import_module(".runner", __name__)


def _runner_proxy() -> ModuleType:
    proxy = ModuleType(f"{__name__}._runner_compat")
    proxy.__getattr__ = lambda name: getattr(_load_runner(), name)
    return proxy


_runner_compat = _runner_proxy()

_COMPAT_MODULES = {
    "RL_init": _input,
    "RL_mcts": _runner_compat,
    "csv_contract": _input,
    "flex_assembly": _flex,
    "hex_assembly_state": _state,
    "inverse_contract": _models,
    "inverse_mcts": _rigid,
    "initialization_diagnostics": _input,
    "rigid_diagnostics": _runner_compat,
    "flex_diagnostics": _runner_compat,
    "side_diagnostics": _runner_compat,
    "visualization": _output,
}
for _name, _module in _COMPAT_MODULES.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)


def run_flex_mcts(*args, **kwargs):
    """Run through the rigid and flexible assembly stages."""
    from .runner import run_flex_mcts as _run

    return _run(*args, **kwargs)


def run_side_mcts(*args, **kwargs):
    """Run the complete rigid, flex, and side-chain assembly."""
    from .runner import run_side_mcts as _run

    return _run(*args, **kwargs)


def assemble_from_csv(*args, **kwargs):
    return _load_runner().assemble_from_csv(*args, **kwargs)


def assemble_from_inverse(*args, **kwargs):
    return _load_runner().assemble_from_inverse(*args, **kwargs)


def __getattr__(name):
    if name == "MCTSRunner":
        return _load_runner().MCTSRunner
    raise AttributeError(name)

__all__ = [
    "initialize",
    "initialize_from_inverse",
    "AssemblyConfig",
    "MCTSConfig",
    "HeteroAromaticRequest",
    "MCTSRunner",
    "assemble_from_csv",
    "assemble_from_inverse",
    "run_flex_mcts",
    "run_side_mcts",
    "run_substitution_stage",
]
