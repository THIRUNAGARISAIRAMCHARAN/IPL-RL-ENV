import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def plot_training_results():
    plots_dir = os.path.join(_REPO_ROOT, "training", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Read rewards CSV
    csv_path = os.path.join(_REPO_ROOT, "training", "logs", "rewards.csv")
    episodes = []
    rewards = []

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            ep_rewards: dict[int, list[float]] = {}
            for row in reader:
                ep = int(row.get("episode", 0))
                r = float(row.get("TOTAL", 0))
                if ep not in ep_rewards:
                    ep_rewards[ep] = []
                ep_rewards[ep].append(r)
            for ep in sorted(ep_rewards.keys()):
                episodes.append(ep)
                rewards.append(sum(ep_rewards[ep]) / len(ep_rewards[ep]))

    # Plot 1: Reward Curve
    plt.figure(figsize=(10, 5))
    if episodes and rewards:
        plt.plot(episodes, rewards, color="blue", linewidth=2, label="Trained Agent")
        baseline = [0.5] * len(episodes)
        plt.plot(episodes, baseline, color="gray", linestyle="--", linewidth=1, label="Random Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("IPL RL Agent — Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "reward_curve.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(plots_dir, 'reward_curve.png')}")

    # Plot 2: Win Rate
    plt.figure(figsize=(10, 5))
    if rewards:
        win_rate = [min(r, 1.0) for r in rewards]
        plt.plot(episodes, win_rate, color="green", linewidth=2, label="Win Rate")
        plt.axhline(y=0.5, color="gray", linestyle="--", label="Baseline")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("IPL RL Agent — Win Rate vs Episodes")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "win_rate.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(plots_dir, 'win_rate.png')}")

    # Plot 3: Before vs After comparison
    if len(rewards) >= 10:
        early = rewards[:10]
        late = rewards[-10:]
        plt.figure(figsize=(8, 5))
        plt.bar(
            ["Early (ep 1-10)", "Late (last 10)"],
            [sum(early) / len(early), sum(late) / len(late)],
            color=["#ff6b6b", "#51cf66"],
        )
        plt.ylabel("Average Reward")
        plt.title("Before vs After Training")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "before_after.png"), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(plots_dir, 'before_after.png')}")


if __name__ == "__main__":
    plot_training_results()
