import os
import json
import csv
import threading
from collections import defaultdict
from typing import Any


class RewardLogger:
    FILES = {
        "rewards": "training/logs/rewards.csv",
        "reward_curve": "training/logs/reward_curve.json",
        "squads": "training/logs/squads.json",
        "auction_log": "training/logs/auction_log.json",
        "season": "training/logs/season_results.json",
        "transfer": "training/logs/transfer_log.json",
        "behaviors": "training/logs/behavior_summaries.json",
        "insights": "training/logs/emergent_insights.json",
    }

    REWARDS_HEADER = [
        "episode",
        "team_id",
        "team_name",
        "value_pick",
        "synergy",
        "late_bonus",
        "panic_penalty",
        "block_reward",
        "waste_penalty",
        "balance_bonus",
        "season_total",
        "transfer_total",
        "TOTAL",
        "budget_wasted_cr",
        "final_position",
        "squad_balance_score",
    ]

    def __init__(self, log_path: str | None = None):
        self.event_log_path = log_path or "training/logs/reward_events.jsonl"
        os.makedirs("training/logs", exist_ok=True)
        self._lock = threading.Lock()  # thread-safe writes
        self._init_files()

    def _init_files(self):
        # Create empty versions of all files if not exist
        # rewards.csv: write header row only
        # All .json: write empty {} or []
        for key, path in self.FILES.items():
            if os.path.exists(path):
                continue
            if key == "rewards":
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.REWARDS_HEADER)
                    writer.writeheader()
            elif key in {"season", "squads", "behaviors", "reward_curve"}:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({}, f, indent=2)
            elif key == "insights":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "reward_improvement_pct": 0.0,
                            "win_rate_improvement_pct": 0.0,
                            "budget_efficiency_improvement_pct": 0.0,
                        },
                        f,
                        indent=2,
                    )
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)

    def _read_json(self, path: str, default: Any):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _write_json(self, path: str, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _append_json_list(self, path: str, item: Any):
        data = self._read_json(path, [])
        if not isinstance(data, list):
            data = []
        data.append(item)
        self._write_json(path, data)

    def log_rewards_row(self, row: dict[str, Any]) -> None:
        with self._lock:
            with open(self.FILES["rewards"], "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.REWARDS_HEADER)
                writer.writerow({k: row.get(k, 0) for k in self.REWARDS_HEADER})

    def write(self, episode: int, step: int, reward: float, phase: str, info: dict[str, Any]) -> None:
        with self._lock:
            payload = {
                "episode": episode,
                "step": step,
                "reward": reward,
                "phase": phase,
                "info": info,
            }
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

    def log_episode(
        self,
        episode,
        rewards,
        squads,
        auction_data,
        season_data,
        transfer_data,
        behavior_data,
    ):
        with self._lock:
            # Append 8 rows to rewards.csv (one per team)
            for team_id, reward_row in rewards.items():
                row = {k: reward_row.get(k, 0) for k in self.REWARDS_HEADER}
                row["episode"] = episode
                row["team_id"] = team_id
                row["team_name"] = reward_row.get("team_name", team_id)
                with open(self.FILES["rewards"], "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.REWARDS_HEADER)
                    writer.writerow(row)

            # Overwrite squads.json with latest
            self._write_json(self.FILES["squads"], squads)

            # Append to auction_log.json
            self._append_json_list(self.FILES["auction_log"], {"episode": episode, "data": auction_data})

            # Overwrite season_results.json
            self._write_json(self.FILES["season"], season_data)

            # Append to transfer_log.json
            self._append_json_list(self.FILES["transfer"], {"episode": episode, "data": transfer_data})

            # Append to behavior_summaries.json (for Before vs After comparison)
            self._append_json_list(self.FILES["behaviors"], behavior_data)

            self.export_training_curves()
            self._write_json(self.FILES["insights"], self.get_learning_proof())

    def export_training_curves(self):
        # Read rewards.csv
        rows = []
        with open(self.FILES["rewards"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("episode"):
                    rows.append(row)

        teams = defaultdict(lambda: {"episodes": [], "rewards": [], "wins": [], "budget_efficiency": []})

        for row in rows:
            team_id = row["team_id"]
            episode = int(float(row["episode"]))
            total_reward = float(row.get("TOTAL", 0) or 0)
            final_position = float(row.get("final_position", 8) or 8)
            budget_wasted = float(row.get("budget_wasted_cr", 0) or 0)
            budget_eff = max(0.0, 1.0 - (budget_wasted / 90.0))

            teams[team_id]["episodes"].append(episode)
            teams[team_id]["rewards"].append(total_reward)
            teams[team_id]["wins"].append(1.0 if final_position <= 4 else 0.0)
            teams[team_id]["budget_efficiency"].append(budget_eff)

        # Compute rolling avg (window=10) per team: rewards, win_rate, budget_efficiency
        def rolling_avg(values, window=10):
            out = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                seg = values[start : i + 1]
                out.append(sum(seg) / len(seg))
            return out

        payload = {"episodes": sorted({int(float(r["episode"])) for r in rows}), "teams": {}}
        for team_id, stats in teams.items():
            payload["teams"][team_id] = {
                "rewards": rolling_avg(stats["rewards"], 10),
                "win_rate": rolling_avg(stats["wins"], 10),
                "budget_efficiency": rolling_avg(stats["budget_efficiency"], 10),
            }

        # Write reward_curve.json
        # { 'episodes': [1,2,...], 'teams': { 'MI': { 'rewards':[...], 'win_rate':[...], ... } } }
        self._write_json(self.FILES["reward_curve"], payload)

    def get_learning_proof(self) -> dict:
        # Compare first 10 vs last 10 episodes
        curve = self._read_json(self.FILES["reward_curve"], {"teams": {}})
        teams = curve.get("teams", {})
        if not teams:
            return {
                "reward_improvement_pct": 0.0,
                "win_rate_improvement_pct": 0.0,
                "budget_efficiency_improvement_pct": 0.0,
            }

        def pct(first: float, last: float) -> float:
            if abs(first) < 1e-9:
                return 0.0 if abs(last) < 1e-9 else 100.0
            return ((last - first) / abs(first)) * 100.0

        reward_improvements = []
        win_improvements = []
        budget_improvements = []

        for _, data in teams.items():
            rewards = data.get("rewards", [])
            wins = data.get("win_rate", [])
            budgets = data.get("budget_efficiency", [])
            if len(rewards) < 20 or len(wins) < 20 or len(budgets) < 20:
                continue

            first_reward = sum(rewards[:10]) / 10.0
            last_reward = sum(rewards[-10:]) / 10.0
            first_win = sum(wins[:10]) / 10.0
            last_win = sum(wins[-10:]) / 10.0
            first_budget = sum(budgets[:10]) / 10.0
            last_budget = sum(budgets[-10:]) / 10.0

            reward_improvements.append(pct(first_reward, last_reward))
            win_improvements.append(pct(first_win, last_win))
            budget_improvements.append(pct(first_budget, last_budget))

        # Return: { reward_improvement_pct, win_rate_improvement_pct, budget_efficiency_improvement_pct }
        # These 3 numbers prove improvement to judges
        if not reward_improvements:
            return {
                "reward_improvement_pct": 0.0,
                "win_rate_improvement_pct": 0.0,
                "budget_efficiency_improvement_pct": 0.0,
            }
        return {
            "reward_improvement_pct": round(sum(reward_improvements) / len(reward_improvements), 2),
            "win_rate_improvement_pct": round(sum(win_improvements) / len(win_improvements), 2),
            "budget_efficiency_improvement_pct": round(sum(budget_improvements) / len(budget_improvements), 2),
        }

    def log_auction_reward(self, team_id, reward_components):
        with self._lock:
            payload = {"team_id": team_id, "reward_components": reward_components}
            self._append_json_list(self.FILES["auction_log"], payload)


if __name__ == "__main__":
    logger = RewardLogger()
    for fname in RewardLogger.FILES.values():
        assert os.path.exists(fname), f"MISSING: {fname}"
    print("RewardLogger OK - all", len(RewardLogger.FILES), "log files created")
