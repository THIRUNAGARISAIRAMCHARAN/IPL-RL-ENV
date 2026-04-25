from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


TEAM_ORDER = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    curve_path = root / "training" / "logs" / "reward_curve.json"
    out_path = root / "reward_curve.png"

    if not curve_path.exists():
        raise FileNotFoundError(f"Missing reward curve file: {curve_path}")

    data = json.loads(curve_path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    teams = data.get("teams", {})

    plt.figure(figsize=(11, 6))
    plotted = 0
    for team in TEAM_ORDER:
        series = teams.get(team, {}).get("rewards", [])
        if not series:
            continue
        x = episodes[: len(series)] if episodes else list(range(1, len(series) + 1))
        plt.plot(x, series, label=team)
        plotted += 1

    if plotted == 0:
        # Keep a valid output artifact even with empty logs.
        plt.text(0.5, 0.5, "No reward curve data yet", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    plt.title("IPL RL Reward Curves by Team")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.25)
    if plotted > 0:
        plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved reward curve plot to {out_path}")


if __name__ == "__main__":
    main()
