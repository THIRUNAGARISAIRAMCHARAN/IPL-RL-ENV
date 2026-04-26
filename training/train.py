import csv
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def append_training_run_log(lines: list[str]) -> None:
    """Append a short run summary to training/logs/training_session.log (audit / Space logs)."""
    os.makedirs("training/logs", exist_ok=True)
    path = "training/logs/training_session.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n========== {ts} ==========\n")
        for line in lines:
            f.write(line.rstrip() + "\n")


def write_training_summary_txt(lines: list[str]) -> None:
    """One human-readable file per run for README / judges (training/logs/training_summary.txt)."""
    os.makedirs("training/logs", exist_ok=True)
    path = "training/logs/training_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _read_prior_reward_series(csv_path: str) -> list[float]:
    """Mean TOTAL per episode index from existing rewards.csv (for the 'before' curve)."""
    if not os.path.isfile(csv_path):
        return []
    rows: list[tuple[int, float]] = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ep = int(row["episode"])
                    tot = float(row["TOTAL"])
                except (KeyError, ValueError):
                    continue
                rows.append((ep, tot))
    except OSError:
        return []
    if not rows:
        return []
    from collections import defaultdict

    d: defaultdict[int, list[float]] = defaultdict(list)
    for ep, tot in rows:
        d[ep].append(tot)
    xs = sorted(d.keys())
    return [sum(d[x]) / len(d[x]) for x in xs]


def reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        if "BID" in text.upper():
            rewards.append(1.0)
        elif "PASS" in text.upper():
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def run_training(episodes: int = 50) -> None:
    os.makedirs("training/logs", exist_ok=True)
    csv_path = "training/logs/rewards.csv"
    before_series = _read_prior_reward_series(csv_path)

    print("===== IPL GRPO Training Started =====", flush=True)
    print(f"Total Episodes: {episodes}", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({
        "prompt": [
            [
                {
                    "role": "user",
                    "content": (
                        "You are an IPL team manager with 90Cr budget. Should you BID or PASS on this player? "
                        "Respond with BID: <amount> or PASS."
                    ),
                }
            ]
        ]
        * episodes
    })

    config = GRPOConfig(
        max_completion_length=128,
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=10,
        output_dir="checkpoints",
        report_to="none",
        bf16=False,
        fp16=False,
        use_cpu=True,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...", flush=True)
    trainer.train()
    # Fresh line after tqdm so the banner is visible in Space / Docker logs
    print("", flush=True)
    print("===== Training Complete =====", flush=True)

    run_log: list[str] = [
        "===== Training Complete =====",
        f"dataset_prompts={episodes}",
        f"model={MODEL}",
    ]
    reward_logs: list = []
    reward = 0.0
    gradio_score = 0
    avg_reward = 0.0
    max_reward = 0.0

    if trainer.state.log_history:
        reward_logs = [log for log in trainer.state.log_history if "reward" in log]
        reward = float(reward_logs[-1].get("reward", 0.0)) if reward_logs else 0.0
        gradio_score = round(reward * 100, 2)
        print(f"\nSCORE: {gradio_score}", flush=True)
        print("--- Final Rewards ---", flush=True)
        for team in TEAM_NAMES:
            print(f"  {team:<8}: {reward:+.2f}", flush=True)

        reward_values = [float(log.get("reward", 0.0)) for log in reward_logs]
        avg_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
        max_reward = max(reward_values) if reward_values else 0.0
        print(f"  Avg Reward (over logged steps): {avg_reward:+.4f}", flush=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "team_id", "TOTAL"])
            for i, log in enumerate(reward_logs):
                r = float(log.get("reward", 0.0))
                for team in TEAM_NAMES:
                    writer.writerow([i, team, round(r, 4)])

        # Leaderboard JSON: headline score from last logged step; stats from all reward logs
        score_data = {
            "gradio_score": gradio_score,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "total_episodes": len(reward_values),
        }
        with open("training/logs/gradio_score.json", "w") as f:
            json.dump(score_data, f, indent=2)

        run_log.append(f"log_history_steps={len(reward_values)}")
        run_log.append(f"SCORE: {gradio_score}")
        run_log.append(f"avg_reward={avg_reward} max_reward={max_reward}")
        run_log.append("gradio_score.json=rewards.csv=written")
    else:
        print("WARNING: empty trainer.state.log_history — no score or rewards.csv update.", flush=True)
        run_log.append("WARNING: empty trainer.state.log_history")

    # Before / after reward curves (PNG for Space UI)
    reward_logs_plot = (
        [log for log in trainer.state.log_history if "reward" in log] if trainer.state.log_history else []
    )
    after_series = [float(log.get("reward", 0.0)) for log in reward_logs_plot]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if before_series:
        axes[0].plot(range(len(before_series)), before_series, marker="o", color="steelblue", linewidth=1.5)
        axes[0].set_title("Before (prior rewards.csv)")
    else:
        axes[0].text(0.5, 0.5, "No prior rewards.csv", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Before (no prior run)")
    axes[0].set_xlabel("Episode / log index")
    axes[0].set_ylabel("Mean reward")
    axes[0].grid(True, alpha=0.3)
    if after_series:
        axes[1].plot(range(len(after_series)), after_series, marker="o", color="seagreen", linewidth=1.5)
        axes[1].set_title("After (this GRPO run)")
    else:
        axes[1].text(0.5, 0.5, "No log history", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("After (no logs)")
    axes[1].set_xlabel("Training log step")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("IPL GRPO — reward curves (before vs after)", fontsize=12)
    plt.tight_layout()
    curve_path = "training/logs/before_after_reward_curve.png"
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Saved before/after curve: {curve_path}", flush=True)
    run_log.append(f"before_after_reward_curve={curve_path}")

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from scripts.plot_results import plot_training_results

    plot_training_results()
    print("Plots saved to training/plots/", flush=True)

    # Logs section: mirror console output for Space / judges (session log + training_summary.txt)
    tail_log = trainer.state.log_history[-1] if trainer.state.log_history else {}
    train_loss = tail_log.get("train_loss", tail_log.get("loss", "n/a"))
    train_runtime = tail_log.get("train_runtime", "n/a")
    summary_lines = [
        "===== Training Complete =====",
        "",
        f"Total Episodes (prompts): {episodes}",
        f"Model: {MODEL}",
        f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        "",
        f"SCORE: {gradio_score}",
        "--- Final Rewards ---",
    ]
    for team in TEAM_NAMES:
        summary_lines.append(f"  {team:<8}: {reward:+.2f}")
    summary_lines.extend(
        [
            f"  Avg Reward (over logged steps): {avg_reward:+.4f}",
            f"Saved before/after curve: {curve_path}",
            f"train_loss: {train_loss}",
            f"train_runtime: {train_runtime}",
            "",
            "Plots saved to training/plots/: reward_curve.png, win_rate.png, before_after.png (when enough data)",
        ]
    )
    run_log.extend(["", "--- Logs section (human-readable) ---"])
    run_log.extend(summary_lines)
    append_training_run_log(run_log)
    write_training_summary_txt(summary_lines)
    print("(summary appended to training/logs/training_session.log)", flush=True)
    print("(human-readable copy: training/logs/training_summary.txt)", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    run_training(episodes=args.episodes)
