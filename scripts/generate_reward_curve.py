import json
import os
import matplotlib.pyplot as plt

def generate_plot():
    log_path = "training/logs/reward_curve.json"
    output_path = "reward_curve.png"

    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        teams = data.get("teams", {})

        plt.figure(figsize=(12, 7))
        
        for team, metrics in teams.items():
            rewards = metrics.get("rewards", [])
            if rewards:
                # Plot with markers for better visibility on small versions
                plt.plot(episodes[:len(rewards)], rewards, label=team, linewidth=2)

        plt.title("IPL RL Training: Reward Progression by Team", fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Team Reward", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(output_path, dpi=150)
        print(f"Reward curve saved to {output_path}")

    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    generate_plot()
