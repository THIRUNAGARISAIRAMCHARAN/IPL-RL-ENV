from __future__ import annotations

import argparse
import os
import re
import sys
from statistics import mean

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

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
Budget: Rs.{obs.get('own_budget', 90):.1f} Cr  Squad: {len(obs.get('own_squad', []))} players
Player: {obs.get('current_player', {}).get('role', '?')} {obs.get('current_player', {}).get('tier', '?')}
Current bid: Rs.{obs.get('current_bid', 0):.1f} Cr  Remaining: {obs.get('players_remaining', 0)}
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


def _load_quantized_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            load_in_4bit=True,
            device_map="auto",
        )
        print(f"Loaded {MODEL} in 4-bit mode.")
    except Exception:
        model_name = FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                FALLBACK_MODEL,
                load_in_4bit=True,
                device_map="auto",
            )
            print(f"Loaded fallback {FALLBACK_MODEL} in 4-bit mode.")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(FALLBACK_MODEL).to(device)
            print(f"4-bit unavailable, loaded {FALLBACK_MODEL} in standard precision.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model_name, model, tokenizer


def _build_reward_rows(env: IPLAuctionEnv, episode: int):
    rows = {}
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


def run_episode(model, tokenizer, env, logger, ep_num):
    obs = env.reset()
    done = False
    all_rewards = []
    device = model.device if hasattr(model, "device") else "cpu"

    while not done:
        actions = {}
        for team_name in TEAM_NAMES:
            prompt = obs_to_prompt(obs.get(team_name, {}), team_name)
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            attn = tokens["attention_mask"].to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = out[0, input_ids.shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            actions[team_name] = parse_action(text)

        obs, rewards, done, _info = env.step(actions)
        all_rewards.extend([float(rewards.get(t, 0.0)) for t in TEAM_NAMES])

    rows = _build_reward_rows(env, ep_num)
    logger.log_episode(
        episode=ep_num,
        rewards=rows,
        squads=env.team_squads,
        auction_data=env.auction_engine.auction_log if env.auction_engine else [],
        season_data=env.last_season_results,
        transfer_data=env.transfer_market.trade_log if env.transfer_market else [],
        behavior_data=env.get_info().get("behavior_summaries", {}),
    )
    return all_rewards


def main():
    parser = argparse.ArgumentParser(description="Train IPL RL on HF Space T4 Medium.")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    episodes = max(200, int(args.episodes))

    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)
        print("Logged in to HuggingFace using HF_TOKEN.")
    else:
        print("HF_TOKEN not found, proceeding without explicit login.")

    wandb_run = None
    if os.getenv("WANDB_API_KEY"):
        try:
            import wandb

            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb_run = wandb.init(project="ipl-rl-env", name="hf-t4-train")
            print("wandb logging enabled.")
        except Exception:
            print("wandb init failed; continuing without wandb.")

    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)
    model_name, model, tokenizer = _load_quantized_model()
    env = IPLAuctionEnv()
    logger = RewardLogger()

    print(f"Training model {model_name} for {episodes} episodes.")
    for episode in range(episodes):
        rewards = run_episode(model, tokenizer, env, logger, episode)
        avg_reward = mean(rewards) if rewards else 0.0
        best_reward = max(rewards) if rewards else 0.0
        print(f"Ep {episode:4d} | Avg: {avg_reward:+.2f} | Best: {best_reward:+.2f}")

        if wandb_run is not None:
            wandb_run.log({"episode": episode, "avg_reward": avg_reward, "best_reward": best_reward})

        if episode % 100 == 0 and episode > 0:
            ckpt = os.path.join(ROOT_DIR, "checkpoints", f"ep_{episode}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint: {ckpt}")

    if wandb_run is not None:
        wandb_run.finish()
    print("HF training complete.")


if __name__ == "__main__":
    main()
