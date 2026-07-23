"""Lightweight source-tree imports for utilities from the local VERL checkout."""

from __future__ import annotations

import os
import sys
import types

VERL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verl_rl"
)


def install_verl_stubs() -> None:
    """Expose ``verl`` packages without executing VERL's top-level initializer."""
    verl_pkg = os.path.join(VERL_ROOT, "verl")
    if not os.path.isdir(verl_pkg):
        raise RuntimeError(
            f"verl submodule not found at {VERL_ROOT}; run `git submodule update --init`."
        )

    for name, path in (
        ("verl", verl_pkg),
        ("verl.utils", os.path.join(verl_pkg, "utils")),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [path]
            sys.modules[name] = module
    sys.modules["verl"].utils = sys.modules["verl.utils"]


__all__ = ["VERL_ROOT", "install_verl_stubs"]
