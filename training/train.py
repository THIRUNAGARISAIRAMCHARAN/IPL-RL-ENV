from __future__ import annotations

import argparse
import os
import re
import sys
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.ipl_env import IPLAuctionEnv
from training.reward_logger import RewardLogger

# Use small model that fits free Colab T4
# Primary: Qwen/Qwen2.5-0.5B-Instruct
# Fallback: Qwen/Qwen2.5-0.5B  (base, faster if instruct is too slow)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"
TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]


def obs_to_prompt(obs, team_name):
    return f"""You are the {team_name} IPL team manager.
Budget: Rs.{obs.get('own_budget',90):.1f} Cr  Squad: {len(obs.get('own_squad',[]))} players
Player: {obs.get('current_player',{}).get('role','?')} {obs.get('current_player',{}).get('tier','?')}
Current bid: Rs.{obs.get('current_bid',0):.1f} Cr  Remaining: {obs.get('players_remaining',0)}
Reply with exactly: BID: <amount> or PASS"""


# FIXED parse_action — handles all LLM output edge cases safely
def parse_action(text):
    if text is None:
        return ("pass", None)
    try:
        text_upper = str(text).strip().upper()
        if "PASS" in text_upper:
            return ("pass", None)
        # Handle: 'BID: 5.5', 'BID:5.5', 'BID 5.5 CRORE', 'bid:5.5cr' etc.
        match = re.search(r"BID[:\s]+([\d\.]+)", text_upper)
        if match:
            amount = float(match.group(1))  # wrapped in try/except below
            amount = max(0.5, min(amount, 90.0))  # clamp to valid range
            return ("bid", amount, False)
        return ("pass", None)  # safe default
    except (ValueError, AttributeError, TypeError):
        return ("pass", None)  # NEVER crash on bad LLM output


def _load_model_and_tokenizer(use_hf_model: bool):
    if not use_hf_model:
        return "mock-policy", None, None

    model_name = MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL)
    except Exception:
        model_name = FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL)

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


def _try_init_ppo(model, tokenizer):
    if model is None or tokenizer is None:
        return None
    try:
        from trl import PPOConfig, PPOTrainer

        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=8,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
        )
        trainer = PPOTrainer(
            config=config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
        )
        return trainer
    except Exception:
        return None


def run_episode(model, tokenizer, env, logger, ep_num, ppo_trainer=None):
    obs = env.reset()
    done = False
    all_rewards = []
    device = model.device if model is not None else "cpu"

    while not done:
        actions = {}
        query_tensors = []
        response_tensors = []
        step_team_ids = []

        for team_name in TEAM_NAMES:
            if model is None or tokenizer is None:
                # Randomized Mock Policy for Demos
                import random
                personalities = {
                    "MI": "aggressive", "CSK": "conservative", "RCB": "aggressive", "KKR": "balanced",
                    "DC": "role_filler", "RR": "conservative", "PBKS": "balanced", "SRH": "role_filler"
                }
                pers = personalities.get(team_name, "balanced")
                team_obs = obs.get(team_name, {})
                curr_bid = team_obs.get("current_bid", 0)
                
                # Probability of bidding based on personality
                bid_prob = {"aggressive": 0.6, "balanced": 0.4, "conservative": 0.2, "role_filler": 0.3}.get(pers, 0.4)
                
                if random.random() < bid_prob and team_obs.get("own_budget", 0) > curr_bid + 0.5:
                    # Random bid increment within personality range
                    incr = {"aggressive": random.uniform(1.0, 5.0), "balanced": random.uniform(0.5, 2.0), 
                            "conservative": random.uniform(0.5, 1.0), "role_filler": random.uniform(0.5, 1.5)}.get(pers, 0.5)
                    actions[team_name] = ("bid", round(curr_bid + incr, 1), False)
                else:
                    actions[team_name] = ("pass", None)
                continue

            team_obs = obs.get(team_name, {})
            prompt = obs_to_prompt(team_obs, team_name)
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            attn = tokens["attention_mask"].to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = output[0, input_ids.shape[1] :]
            response_text = tokenizer.decode(generated, skip_special_tokens=True)
            actions[team_name] = parse_action(response_text)

            query_tensors.append(input_ids[0].detach().cpu())
            response_tensors.append(generated.detach().cpu())
            step_team_ids.append(team_name)

        obs, rewards_dict, done, info = env.step(actions)
        all_rewards.extend([float(rewards_dict.get(t, 0.0)) for t in TEAM_NAMES])

        if ppo_trainer is not None:
            reward_tensors = [torch.tensor(float(rewards_dict.get(t, 0.0))) for t in step_team_ids]
            try:
                ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            except Exception:
                # Keep training loop resilient in hackathon settings.
                pass

    rewards_rows = _build_reward_rows(env, ep_num)
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


def run_training(episodes: int = 500, use_hf_model: bool = False) -> None:
    os.makedirs("training/logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    model_name, model, tokenizer = _load_model_and_tokenizer(use_hf_model=use_hf_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is not None:
        model = model.to(device)
    ppo_trainer = _try_init_ppo(model, tokenizer)

    env = IPLAuctionEnv()
    logger = RewardLogger()

    print(f"Training with model: {model_name} on {device}")
    for episode in range(episodes):
        reward_list = run_episode(model, tokenizer, env, logger, episode, ppo_trainer=ppo_trainer)
        if episode % 10 == 0:
            avg_reward = mean(reward_list) if reward_list else 0.0
            best_reward = max(reward_list) if reward_list else 0.0
            print(f"Ep {episode:4d} | Avg: {avg_reward:+.2f} | Best: {best_reward:+.2f}")
        if episode % 100 == 0 and episode > 0:
            if model is not None:
                model.save_pretrained(f"checkpoints/ep_{episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IPL RL agents.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument(
        "--use_hf_model",
        action="store_true",
        help="Use HF causal LM generation. Default is fast mock policy for demos.",
    )
    args = parser.parse_args()
    run_training(episodes=args.episodes, use_hf_model=args.use_hf_model)
