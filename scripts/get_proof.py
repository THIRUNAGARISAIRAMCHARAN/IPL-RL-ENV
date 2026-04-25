from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.reward_logger import RewardLogger


def main() -> None:
    proof = RewardLogger().get_learning_proof()
    print("Copy-paste for BLOG.md:")
    print(f"Reward +{proof.get('reward_improvement_pct', 0):.2f}%")
    print(f"Win Rate +{proof.get('win_rate_improvement_pct', 0):.2f}%")
    print(f"Budget Efficiency +{proof.get('budget_efficiency_improvement_pct', 0):.2f}%")


if __name__ == "__main__":
    main()
