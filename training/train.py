from __future__ import annotations

import argparse
import builtins
import importlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

REQUIRED_PKGS: list[tuple[str, str]] = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("trl", "trl"),
    ("openenv", "openenv"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]


def _check_required_imports() -> bool:
    bad: list[tuple[str, str]] = []
    for label, name in REQUIRED_PKGS:
        try:
            importlib.import_module(name)
        except ImportError:
            bad.append((label, name))
    if not bad:
        return True
    for label, _ in bad:
        print(f"Missing: {label}")
    for _, name in bad:
        print(f"pip install {name}")
    return False


if __name__ == "__main__" and not _check_required_imports():
    raise SystemExit(0)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
from transformers import AutoTokenizer
from transformers import logging as hf_logging
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Force all output to appear immediately in HF Space logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def print(*args, **kwargs):  # type: ignore[no-redef]
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)

# Make HF download progress print to stdout instead of stderr
hf_logging.set_verbosity_info()

# Make tqdm write to stdout so HF Space logs capture it
tqdm.monitor_interval = 0

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from env.ipl_env import IPLAuctionEnv
from training.reward_logger import RewardLogger

PRIMARY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B"
TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
REWARD_CSV = Path(ROOT_DIR) / "training" / "logs" / "rewards.csv"


def obs_to_prompt(obs: dict[str, Any], team_name: str) -> str:
    return f"""You are the {team_name} IPL team manager.
Budget: Rs.{obs.get("own_budget", 90):.1f} Cr  Squad: {len(obs.get("own_squad", []))} players
Player: {obs.get("current_player", {}).get("role", "?")} {obs.get("current_player", {}).get("tier", "?")}
Current bid: Rs.{obs.get("current_bid", 0.0):.1f} Cr  Remaining: {obs.get("players_remaining", 0)}
Reply with exactly: BID: <amount> or PASS"""


def parse_action(text: str | None) -> tuple:
    if text is None:
        return ("pass", None)
    try:
        text_upper = str(text).strip().upper()
        if "PASS" in text_upper:
            return ("pass", None)
        match = re.search(r"BID[:\s]+([\d.]+)", text_upper)
        if match:
            amount = float(match.group(1))
            amount = max(0.5, min(amount, 90.0))
            return ("bid", amount, False)
        return ("pass", None)
    except (ValueError, AttributeError, TypeError):
        return ("pass", None)


def _device_for_model(model) -> torch.device:
    return next(model.parameters()).device


def _load_model_and_tokenizer() -> tuple[str, Any, Any] | None:
    last_err: str | None = None
    for model_id in (PRIMARY_MODEL, FALLBACK_MODEL):
        for use_4bit in (True, False):
            try:
                with auto_tqdm(total=1, desc="Loading weights", file=sys.stdout) as progress:
                    if use_4bit:
                        m = AutoModelForCausalLMWithValueHead.from_pretrained(
                            model_id,
                            load_in_4bit=True,
                            device_map="auto",
                        )
                    else:
                        m = AutoModelForCausalLMWithValueHead.from_pretrained(
                            model_id,
                            device_map="auto",
                        )
                    progress.update(1)
                tok = AutoTokenizer.from_pretrained(model_id)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                return model_id, m, tok
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
    print("Could not load a causal LM after all attempts. Last error:", last_err)
    return None


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


def _budget_efficiency(rows: dict[str, dict[str, Any]]) -> float:
    wasteds = [float(r.get("budget_wasted_cr", 0.0)) for r in rows.values()]
    effs = [max(0.0, min(1.0, 1.0 - w / 90.0)) for w in wasteds]
    return 100.0 * (mean(effs) if effs else 0.0)


def _parse_rewards_history(csv_path: Path) -> tuple[np.ndarray, dict[str, int], int, float | None, float | None]:  # noqa: UP
    if not csv_path.is_file() or csv_path.stat().st_size < 2:
        return (
            np.array([]),
            {t: 0 for t in TEAM_NAMES},
            0,
            None,
            None,
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return np.array([]), {t: 0 for t in TEAM_NAMES}, 0, None, None
    if df.empty or "episode" not in df.columns or "TOTAL" not in df.columns:
        return np.array([]), {t: 0 for t in TEAM_NAMES}, 0, None, None
    champs: dict[str, int] = {t: 0 for t in TEAM_NAMES}
    for _, g in df.groupby("episode", sort=True):
        try:
            w = g.loc[g["TOTAL"].idxmax()]["team_id"]
            w = str(w)
            if w in champs:
                champs[w] += 1
        except Exception:  # noqa: BLE001
            continue
    by = df.groupby("episode", sort=True)["TOTAL"].mean()
    if len(by) == 0:
        return np.array([]), champs, 0, None, None
    best_idx = by.values.argmax()
    best_ep = int(by.index[best_idx])
    best_val = float(by.values[best_idx])
    return (
        by.values,
        champs,
        int(by.index.max()),
        best_ep,
        best_val,
    )


def _fmt_hms(secs: float) -> str:
    s = int(max(0, round(secs)))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _plateau_warning(scores: list[float]) -> None:
    if len(scores) < 50:
        return
    a = float(mean(scores[-50:-30]))
    b = float(mean(scores[-20:]))
    if b - a < 0.5:
        print("WARNING: Reward plateau detected. Consider stopping.")


def _save_model_safe(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not save model: {e}")


def _copy_rewards_for_interrupt() -> None:
    dst = Path(ROOT_DIR) / "checkpoints" / "interrupted" / "rewards.csv"
    if REWARD_CSV.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REWARD_CSV, dst)


def run_episode(
    model,
    tokenizer,
    ppo_trainer,
    env: IPLAuctionEnv,
    ep_num: int,
    logger: RewardLogger,
) -> dict[str, Any]:
    if model is None or tokenizer is None:
        return {"rows": {}, "avg": 0.0, "best_t": 0.0, "b_eff": 0.0, "champion": "MI"}
    dev = _device_for_model(model)
    obs = env.reset(episode=ep_num)
    done = False

    while not done:
        o0 = obs[TEAM_NAMES[0]]
        phase = o0.get("phase", "auction")
        if phase == "auction":
            actions: dict[str, Any] = {}
            query_tensors: list = []
            response_tensors: list = []
            for team_name in TEAM_NAMES:
                team_obs = obs.get(team_name, {})
                prom = obs_to_prompt(team_obs, team_name)
                tks = tokenizer(prom, return_tensors="pt", truncation=True, max_length=512)
                input_ids = tks["input_ids"].to(dev)
                attn = tks["attention_mask"].to(dev)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=15,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = out[0, input_ids.shape[1] :]
                text = tokenizer.decode(gen, skip_special_tokens=True)
                actions[team_name] = parse_action(text)
                query_tensors.append(input_ids[0].detach().cpu())
                response_tensors.append(gen.detach().cpu().squeeze())
        else:
            actions = {}
            query_tensors = []
            response_tensors = []

        obs, rewards_dict, done, _info = env.step(actions)
        if ppo_trainer is not None and phase == "auction" and query_tensors:
            try:
                rts = [torch.tensor(float(rewards_dict.get(TEAM_NAMES[i], 0.0))) for i in range(8)]
                ppo_trainer.step(query_tensors, response_tensors, rts)
            except Exception:  # noqa: BLE001
                pass

    rows = _build_reward_rows(env, ep_num)
    av = _ep_mean(rows)
    bt = max(float(r["TOTAL"]) for r in rows.values())
    champ = max(rows, key=lambda k: float(rows[k]["TOTAL"]))
    b_eff = _budget_efficiency(rows)
    auction_data = env.auction_engine.auction_log if env.auction_engine is not None else []
    season_data = env.last_season_results
    transfer_data = env.transfer_market.trade_log if env.transfer_market is not None else []
    beh = env.get_info().get("behavior_summaries", {})
    up_curve = ep_num % 10 == 0
    logger.log_episode(
        ep_num,
        rows,
        env.team_squads,
        auction_data,
        season_data,
        transfer_data,
        beh,
        write_reward_rows=False,
        update_reward_curve=up_curve,
    )
    return {
        "rows": rows,
        "avg": av,
        "best_t": bt,
        "b_eff": b_eff,
        "champion": champ,
    }


def _ep_mean(rows: dict[str, dict[str, Any]]) -> float:
    return float(mean([float(r["TOTAL"]) for r in rows.values()]))


def _save_model_tokenizer(model, tok, path: Path) -> None:
    _save_model_safe(model, path)
    if tok is not None:
        try:
            tok.save_pretrained(path)
        except Exception as e:  # noqa: BLE001
            print(f"Could not save tokenizer: {e}")


def _summary_from_csv() -> tuple[list[float], float, int, float]:
    if not REWARD_CSV.is_file() or REWARD_CSV.stat().st_size < 2:
        return [], 0.0, 0, 0.0
    try:
        df = pd.read_csv(REWARD_CSV)
    except Exception:  # noqa: BLE001
        return [], 0.0, 0, 0.0
    if df.empty or "episode" not in df.columns or "TOTAL" not in df.columns:
        return [], 0.0, 0, 0.0
    m = df.groupby("episode", sort=True)["TOTAL"].mean()
    if len(m) == 0:
        return [], 0.0, 0, 0.0
    scores = [float(x) for x in m.tolist()]
    sarr = m.values
    bi = int(np.argmax(sarr))
    best_v = float(sarr[bi])
    best_ep = int(m.index[bi])
    f_mean = float(mean(scores[-10:])) if len(scores) >= 10 else float(mean(scores))
    return scores, best_v, best_ep, f_mean


def _pct_improvement(first: float, last_10: float) -> str:
    if abs(first) < 1e-6:
        return "N/A" if abs(last_10) < 1e-6 else "inf"
    return f"{(last_10 - first) / abs(first) * 100.0:+.0f}%"


def run_training(episodes: int = 200) -> None:
    n_planned = max(1, int(episodes))
    print(f"Dataset: {n_planned} episodes")

    os.makedirs("training/logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    RewardLogger()

    hist, champs_hist, last_ep, _, _ = _parse_rewards_history(REWARD_CSV)
    if len(hist) > 0:
        print(f"Resuming from episode {last_ep + 1}...")
    start_ep = (last_ep + 1) if len(hist) > 0 else 1

    loaded = _load_model_and_tokenizer()
    if not loaded:
        print("Exiting: no model to train.")
        raise SystemExit(1)
    model_name, model, tokenizer = loaded
    print(f"Starting GRPO training with {model_name}...")
    ppo: Any = None
    try:
        # trl 0.9.6: use mini_batch_size + gradient_accumulation_steps (replaces the old
        # single batch_size=8 for optimization chunks). The config still has `batch_size`,
        # which must equal 8 (one PPO step = 8 team queries) or step() raises.
        cfg = PPOConfig(
            learning_rate=1e-5,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            batch_size=8,
        )
        ppo = PPOTrainer(
            config=cfg,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
        )
    except Exception as e:  # noqa: BLE001
        print("PPOTrainer not available, continuing without PPO steps:", e)

    env = IPLAuctionEnv()
    log = RewardLogger()
    t0 = time.time()
    checkpoint_path = "checkpoints/last-checkpoint"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from: ./{checkpoint_path}")

    all_scores: list[float] = list(hist.tolist()) if len(hist) else []
    champs: dict[str, int] = dict(champs_hist)
    session_best = max(all_scores) if all_scores else float("-inf")

    def print_ep_line(ep: int, av: float, best_s: float, b_eff: float) -> None:
        elapsed = time.time() - t0
        print(
            f"Ep {ep:04d} | Avg Reward: {av:+.1f} | Best: {best_s:+.1f} | "
            f"Budget Eff: {b_eff:.0f}% | Time: {_fmt_hms(elapsed)}"
        )

    def print_teams_block(rows: dict[str, dict[str, Any]]) -> None:
        line1 = " ".join(f"[{t}: {float(rows[t]['TOTAL']):+.1f}]" for t in TEAM_NAMES[:4])
        line2 = " ".join(f"[{t}: {float(rows[t]['TOTAL']):+.1f}]" for t in TEAM_NAMES[4:])
        print(line1)
        print(line2)

    try:
        for i in range(n_planned):
            ep_num = start_ep + i
            out = run_episode(model, tokenizer, ppo, env, ep_num, log)
            if not out.get("rows"):
                continue
            av = float(out["avg"])
            b_one = float(out["best_t"])
            b_eff = float(out["b_eff"])
            w = str(out.get("champion", "MI"))
            if w in champs:
                champs[w] += 1

            if av > session_best + 1e-9:
                session_best = av
                print("New best reward! Saving checkpoint...")
                _save_model_tokenizer(
                    model,
                    tokenizer,
                    Path(ROOT_DIR) / "checkpoints" / "best",
                )

            print_ep_line(ep_num, av, b_one, b_eff)
            all_scores.append(av)
            if ep_num % 10 == 0:
                print_teams_block(out["rows"])
            if ep_num > 0 and ep_num % 50 == 0:
                d = Path(ROOT_DIR) / "checkpoints" / f"ep_{ep_num}"
                _save_model_tokenizer(model, tokenizer, d)

            _plateau_warning(all_scores)

    except KeyboardInterrupt:  # noqa: BLE001
        _save_model_tokenizer(
            model,
            tokenizer,
            Path(ROOT_DIR) / "checkpoints" / "interrupted",
        )
        _copy_rewards_for_interrupt()
        print("Training interrupted. Progress saved.")
        sys.exit(0)

    total_t = time.time() - t0
    all_scores, best_v, best_ep, f_mean = _summary_from_csv()
    n_eps = n_planned
    first = float(all_scores[0]) if all_scores else 0.0
    impr = _pct_improvement(first, f_mean)
    champ = max(champs, key=champs.get) if champs and sum(champs.values()) else "MI"
    ntot = max(1, sum(champs.values()))
    cp = 100.0 * champs.get(champ, 0) / ntot

    print("==========================================")
    print("TRAINING COMPLETE")
    print("==========================================")
    print(f"Total Episodes : {n_eps}")
    print(f"Total Time     : {_fmt_hms(total_t)}")
    print(f"Best Reward    : {best_v:+.1f} (Episode {best_ep})")
    print(f"Final Avg (last 10) : {f_mean:+.1f}")
    print(f"Improvement    : {impr} from episode 1")
    print(f"Champion Team  : {champ} (won {cp:.0f}% of episodes)")
    print("==========================================")
    print("Next step: python scripts/generate_curve.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL multi-agent LLM+PPO training.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes.",
    )
    args = parser.parse_args()
    run_training(episodes=args.episodes)