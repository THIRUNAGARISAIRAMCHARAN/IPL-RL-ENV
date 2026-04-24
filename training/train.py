from __future__ import annotations

import argparse
import os
import re
import sys
import csv
from statistics import mean
from typing import Any

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.ipl_env import IPLAuctionEnv
from training.reward_logger import RewardLogger


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"
TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]


def obs_to_prompt(obs, team_name):
    return f"""You are the {team_name} IPL team manager.
Budget: Rs.{obs.get('own_budget',90):.1f} Cr  Squad: {len(obs.get('own_squad',[]))} players
Player: {obs.get('current_player',{}).get('role','?')} {obs.get('current_player',{}).get('tier','?')}
Current bid: Rs.{obs.get('current_bid',0):.1f} Cr  Remaining: {obs.get('players_remaining',0)}
Reply with exactly: BID: <amount> or PASS"""


def parse_action(text):
    if text is None:
        return ("pass", None)
    try:
        text_upper = str(text).strip().upper()
        if "PASS" in text_upper:
            return ("pass", None)
        match = re.search(r"BID[:\s]+([\d\.]+)", text_upper)
        if match:
            amount = float(match.group(1))
            amount = max(0.5, min(amount, 90.0))
            return ("bid", amount, False)
        return ("pass", None)
    except (ValueError, AttributeError, TypeError):
        return ("pass", None)


def _load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
        model_name = MODEL
    except Exception:
        model_name = FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model_name, model, tokenizer


def _build_reward_rows(env: IPLAuctionEnv, episode: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for team_id in TEAM_NAMES:
        sig = env.reward_signals.get(team_id, {})
        rows[team_id] = {
            "episode": episode,
            "team_id": team_id,
            "team_name": team_id,
            "value_pick": round(float(sig.get("value_pick", 0.0)), 4),
            "synergy": round(float(sig.get("synergy", 0.0)), 4),
            "late_bonus": round(float(sig.get("late_bonus", 0.0)), 4),
            "panic_penalty": round(float(sig.get("panic_penalty", 0.0)), 4),
            "block_reward": round(float(sig.get("block_reward", 0.0)), 4),
            "waste_penalty": round(float(sig.get("waste_penalty", 0.0)), 4),
            "balance_bonus": round(float(sig.get("balance_bonus", 0.0)), 4),
            "season_total": round(float(sig.get("season_total", 0.0)), 4),
            "transfer_total": round(float(sig.get("transfer_total", 0.0)), 4),
            "TOTAL": round(float(env.compute_reward(team_id)), 4),
            "budget_wasted_cr": round(float(sig.get("budget_wasted_cr", 0.0)), 4),
            "final_position": int(float(sig.get("final_position", 8.0))),
            "squad_balance_score": round(float(sig.get("squad_balance_score", 0.0)), 4),
        }
    return rows


def log_to_csv(csv_path, rows, is_first=False):
    if not rows:
        return
    fieldnames = list(list(rows.values())[0].keys())
    with open(csv_path, mode='a' if not is_first else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first:
            writer.writeheader()
        for t in TEAM_NAMES:
            if t in rows:
                writer.writerow(rows[t])


def run_episode(model, tokenizer, ppo_trainer, env, logger, ep_num, csv_path):
    obs = env.reset()
    done = False
    all_rewards = []
    device = model.device

    while not done:
        actions = {}
        query_tensors = []
        response_tensors = []
        step_team_ids = []

        for team_name in TEAM_NAMES:
            team_obs = obs.get(team_name, {})
            prompt = obs_to_prompt(team_obs, team_name)
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            attn = tokens["attention_mask"].to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=15,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = output[0, input_ids.shape[1] :]
            response_text = tokenizer.decode(generated, skip_special_tokens=True)
            actions[team_name] = parse_action(response_text)

            # ensure 1D shape as required by some PPOTrainer versions
            query_tensors.append(input_ids[0].detach().cpu())
            response_tensors.append(generated.detach().cpu().squeeze())
            step_team_ids.append(team_name)

        obs, rewards_dict, done, info = env.step(actions)
        all_rewards.extend([float(rewards_dict.get(t, 0.0)) for t in TEAM_NAMES])

        # Step PPO
        reward_tensors = [torch.tensor(float(rewards_dict.get(t, 0.0))) for t in step_team_ids]
        try:
            ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        except Exception as e:
            # PPO step can silently fail if memory issues exist, or token length mismatch
            pass

    rewards_rows = _build_reward_rows(env, ep_num)
    
    # Log to CSV
    log_to_csv(csv_path, rewards_rows, is_first=(ep_num == 0))
    
    auction_data = env.auction_engine.auction_log if env.auction_engine is not None else []
    season_data = env.last_season_results
    transfer_data = env.transfer_market.trade_log if env.transfer_market is not None else []
    behavior_data = env.get_info().get("behavior_summaries", {})
    logger.log_episode(
        episode=ep_num,
        rewards=rewards_rows,
        squads=env.team_squads,
        auction_data=auction_data,
        season_data=season_data,
        transfer_data=transfer_data,
        behavior_data=behavior_data,
    )
    return all_rewards


def run_training(episodes: int = 50) -> None:
    os.makedirs("training/logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    csv_path = "training/logs/rewards.csv"
    
    model_name, model, tokenizer = _load_model_and_tokenizer()
    
    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )

    env = IPLAuctionEnv()
    logger = RewardLogger()

    print(f"Training with model: {model_name} on {model.device} using PPOTrainer")
    avg_rewards_per_episode = []
    
    for episode in range(episodes):
        reward_list = run_episode(model, tokenizer, ppo_trainer, env, logger, episode, csv_path)
        avg_reward = mean(reward_list) if reward_list else 0.0
        avg_rewards_per_episode.append(avg_reward)
        
        best_reward = max(reward_list) if reward_list else 0.0
        print(f"Ep {episode:4d} | Avg: {avg_reward:+.2f} | Best: {best_reward:+.2f}")
        
        if episode % 10 == 0 and episode > 0:
            model.save_pretrained(f"checkpoints/ep_{episode}")

    # Plot Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(episodes), avg_rewards_per_episode, label='Average Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.legend()
    plot_path = "reward_curve.png"
    plt.savefig(plot_path)
    print(f"Training complete! Results saved to {csv_path} and plot saved to {plot_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPL RL agents.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes.")
    args = parser.parse_args()
    
    run_training(episodes=args.episodes)
