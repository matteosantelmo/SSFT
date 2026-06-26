from __future__ import annotations

import os
import sys
import types

VERL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verl_rl"
)


def _install_verl_stubs() -> None:
    """Register stub ``verl`` / ``verl.utils`` packages to skip verl's __init__."""
    verl_pkg = os.path.join(VERL_ROOT, "verl")
    for name, path in (("verl", verl_pkg), ("verl.utils", os.path.join(verl_pkg, "utils"))):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]  # mark as a package so submodules import
            sys.modules[name] = mod
    # link child onto parent so ``verl.utils`` attribute access works
    sys.modules["verl"].utils = sys.modules["verl.utils"]


if not os.path.isdir(os.path.join(VERL_ROOT, "verl")):
    raise RuntimeError(
        f"verl submodule not found at {VERL_ROOT}; run `git submodule update --init`."
    )

_install_verl_stubs()

from verl.utils.reward_score import default_compute_score  # noqa: E402



def verify(data_source, response_text, ground_truth, extra_info=None) -> dict:
    """Score one response, mirroring verl's NaiveRewardManager call convention.

    ``ground_truth`` is passed through raw (the verifiers parse it themselves).
    Always returns a dict with a numeric (or None) ``score`` key; never raises,
    so one bad sample can't abort a long run.
    """
    try:
        res = default_compute_score(
            data_source=data_source,
            solution_str=response_text,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    except Exception as exc:  # noqa: BLE001 - isolate per-sample verifier failures
        return {"score": None, "error": f"{type(exc).__name__}: {exc}"}

    if isinstance(res, dict):
        out = dict(res)
        if "score" in out:
            out["score"] = float(out["score"])
        return out
    return {"score": float(res)}
