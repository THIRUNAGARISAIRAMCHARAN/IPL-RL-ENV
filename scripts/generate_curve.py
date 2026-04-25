from __future__ import annotations

"""Convenience entrypoint. Delegates to scripts/generate_reward_curve.py."""

if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    target = here / "generate_reward_curve.py"
    spec = importlib.util.spec_from_file_location("generate_reward_curve", target)
    if spec is None or spec.loader is None:
        print("Could not load generate_reward_curve module.")
        raise SystemExit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
