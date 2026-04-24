from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

try:
    from openenv import BaseEnvironment, validate
except Exception:
    class BaseEnvironment:  # type: ignore[no-redef]
        pass

    def validate(env):  # type: ignore[no-redef]
        obs = env.reset()
        assert isinstance(obs, dict)
        assert len(obs) == 8

from .auction_engine import AuctionEngine
from .season_simulator import SeasonSimulator
from .transfer_market import TransferMarket
from training.reward_logger import RewardLogger


class EmergentBehaviorDetector:
    def __init__(self):
        self.episode_summaries = []

    def analyze_episode(self, agents, auction_log, season_results):
        del auction_log, season_results
        behaviors = {}
        for team_id, agent in agents.items():
            summary = agent.get_behavior_summary()
            label = self._classify_strategy(summary)
            behaviors[team_id] = {
                **summary,
                "label": label,
                "team": getattr(agent, "team_name", str(team_id)),
            }
        self.episode_summaries.append(behaviors)
        return behaviors

    def _classify_strategy(self, summary):
        if summary.get("overbid_rate", 0) > 0.4:
            return "Reckless Bidder"
        if summary.get("block_rate", 0) > 0.2:
            return "Strategic Blocker"
        if summary.get("patience_score", 0) > 3:
            return "Patient Planner"
        if summary.get("bluff_success_rate", 0) > 0.5:
            return "Deceptive Dealer"
        return "Balanced Bidder"

    def detect_learning_shift(self, early_n=10, late_n=10):
        if len(self.episode_summaries) < early_n + late_n:
            return []

        early = self.episode_summaries[:early_n]
        late = self.episode_summaries[-late_n:]
        metrics = ["overbid_rate", "block_rate", "bluff_success_rate", "patience_score"]
        team_ids = set()
        for ep in self.episode_summaries:
            team_ids.update(ep.keys())

        insights = []
        for team_id in sorted(team_ids):
            early_rows = [ep.get(team_id, {}) for ep in early if team_id in ep]
            late_rows = [ep.get(team_id, {}) for ep in late if team_id in ep]
            if not early_rows or not late_rows:
                continue

            early_label_counts = {}
            late_label_counts = {}
            for row in early_rows:
                lbl = row.get("label", "Balanced Bidder")
                early_label_counts[lbl] = early_label_counts.get(lbl, 0) + 1
            for row in late_rows:
                lbl = row.get("label", "Balanced Bidder")
                late_label_counts[lbl] = late_label_counts.get(lbl, 0) + 1

            early_label = max(early_label_counts, key=early_label_counts.get)
            late_label = max(late_label_counts, key=late_label_counts.get)
            if early_label != late_label:
                total_eps = len(self.episode_summaries)
                insights.append(
                    f"{team_id} evolved from {early_label} to {late_label} over {total_eps} episodes"
                )

            def avg(rows, key):
                vals = [float(r.get(key, 0.0)) for r in rows]
                return sum(vals) / max(1, len(vals))

            early_overbid = avg(early_rows, "overbid_rate")
            late_overbid = avg(late_rows, "overbid_rate")
            if early_overbid - late_overbid > 0.1:
                insights.append(
                    f"Average panic bid rate dropped from {early_overbid*100:.0f}% to {late_overbid*100:.0f}%"
                )

            for metric in metrics:
                early_m = avg(early_rows, metric)
                late_m = avg(late_rows, metric)
                if abs(late_m - early_m) > 0.15:
                    direction = "increased" if late_m > early_m else "decreased"
                    insights.append(
                        f"{team_id} {direction} {metric} from {early_m:.2f} to {late_m:.2f}"
                    )
        return insights

    def generate_story_bullets(self) -> list:
        return self.detect_learning_shift()[:5]


class IPLAuctionEnv(BaseEnvironment):
    PHASE_ORDER = ["auction", "season", "transfer", "done"]
    TEAMS = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]

    def __init__(self, num_teams=8, seed=None):
        self.num_teams = int(num_teams)
        self.random = random.Random(seed if seed is not None else 42)
        self.data_dir = Path(__file__).resolve().parents[1] / "data"
        self.players = json.loads((self.data_dir / "players.json").read_text(encoding="utf-8"))
        self.teams = json.loads((self.data_dir / "teams.json").read_text(encoding="utf-8"))
        self.phase = "auction"
        self.phase_step = 0
        self.done = False
        self.episode_rewards = {team_id: 0.0 for team_id in self.TEAMS[: self.num_teams]}
        self.last_season_results: dict[str, Any] = {}
        self.team_squads: dict[str, list[dict[str, Any]]] = {}
        self.agents: dict[str, Any] = {}
        self.behavior_detector = EmergentBehaviorDetector()
        self.last_behavior_summary: dict[str, Any] = {}
        self.episode_idx = 0
        self.reward_logger = RewardLogger(Path(__file__).resolve().parents[1] / "training" / "logs" / "reward_events.jsonl")
        self.reward_signals: dict[str, dict[str, float]] = {}

        self.auction_engine: AuctionEngine | None = None
        self.season_simulator: SeasonSimulator | None = None
        self.transfer_market: TransferMarket | None = None

    def _player_from_id(self, player_id: int) -> dict[str, Any]:
        return next(p for p in self.players if int(p["id"]) == int(player_id))

    def _build_team_squads(self) -> dict[str, list[dict[str, Any]]]:
        assert self.auction_engine is not None
        squads = {}
        for team_id, state in self.auction_engine.team_states.items():
            squads[team_id] = [self._player_from_id(pid) for pid in state["squad"]]
        return squads

    def _build_transfer_teams(self) -> list[dict[str, Any]]:
        assert self.auction_engine is not None
        teams = []
        for t in self.teams[: self.num_teams]:
            tid = str(t["id"])
            teams.append({"id": tid, "budget_cr": self.auction_engine.team_states[tid]["budget"]})
        return teams

    def reset(self) -> dict:
        self.episode_idx += 1
        self.phase = "auction"
        self.phase_step = 0
        self.done = False
        self.last_season_results = {}
        self.episode_rewards = {team_id: 0.0 for team_id in self.TEAMS[: self.num_teams]}
        self.last_behavior_summary = {}
        self.reward_signals = {
            team_id: {
                "value_pick": 0.0,
                "synergy": 0.0,
                "late_bonus": 0.0,
                "panic_penalty": 0.0,
                "block_reward": 0.0,
                "waste_penalty": 0.0,
                "balance_bonus": 0.0,
                "season_total": 0.0,
                "transfer_total": 0.0,
                "TOTAL": 0.0,
                "budget_wasted_cr": 0.0,
                "final_position": 8.0,
                "squad_balance_score": 0.0,
            }
            for team_id in self.TEAMS[: self.num_teams]
        }

        self.auction_engine = AuctionEngine(self.players, self.teams[: self.num_teams], self.num_teams)
        self.team_squads = {tid: [] for tid in self.TEAMS[: self.num_teams]}
        self.season_simulator = SeasonSimulator([{"id": tid, "squad": squad} for tid, squad in self.team_squads.items()])
        self.transfer_market = TransferMarket(
            self._build_transfer_teams(),
            self.team_squads,
            self.season_simulator,
            mid_season_point=7,
        )
        return {team_id: self.get_observation(team_id) for team_id in self.TEAMS[: self.num_teams]}

    def _role_counts(self, squad: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
        for p in squad:
            role = p.get("role")
            if role in counts:
                counts[role] += 1
        return counts

    def _squad_balance_score(self, squad: list[dict[str, Any]]) -> float:
        counts = self._role_counts(squad)
        score = 0.0
        score += 1.0 if counts["WK"] >= 1 else 0.0
        score += min(1.0, counts["BAT"] / 4.0)
        score += min(1.0, counts["BOWL"] / 4.0)
        score += min(1.0, counts["AR"] / 2.0)
        return round(score / 4.0, 4)

    def auction_reward(self, team_id, player, final_price, squad, opponent_data):
        class _FallbackAgent:
            def value_player(self, player_obs, own_squad):
                del own_squad
                s = player_obs.get("visible_stats", {})
                return (
                    float(s.get("batting_avg", 0.0)) * 0.1
                    + float(s.get("strike_rate", 0.0)) * 0.03
                    + float(s.get("wickets_per_match", 0.0)) * 6.0
                )

        agent = self.agents.get(team_id, _FallbackAgent())
        estimated_value = float(agent.value_player(player, squad))
        price = max(0.01, float(final_price))
        players_remaining = int(opponent_data.get("players_remaining", 0))
        active_bidders = int(opponent_data.get("num_active_bidders", 0))
        role = player.get("role")

        # 1. Value Pick
        value_pick = ((estimated_value / price) - 1.0) * 10.0

        # 2. Synergy
        new_tags = set(player.get("synergy_tags", []))
        synergy_pairs = 0
        for p in squad:
            if int(p.get("id", -1)) == int(player.get("id", -2)):
                continue
            if new_tags.intersection(set(p.get("synergy_tags", []))):
                synergy_pairs += 1
        synergy = 3.0 * synergy_pairs

        # 3. Late Bonus
        fair_bid = abs(price - estimated_value) <= max(0.2, estimated_value * 0.2)
        late_bonus = 8.0 if players_remaining < 30 and fair_bid else 0.0

        # 4. Panic Penalty
        panic_penalty = -12.0 if price > estimated_value * 1.35 and active_bidders >= 4 else 0.0

        # 5. Block Reward
        blocked_any = False
        gaps = opponent_data.get("opponent_role_gaps", {})
        for opp_team, role_gap in gaps.items():
            if opp_team == team_id:
                continue
            if role_gap.get(role, False):
                blocked_any = True
                break
        block_reward = 10.0 if blocked_any and price > float(player.get("base_price_cr", 0.0)) * 1.1 else 0.0

        # 6/7 are end-of-auction signals and are computed once at auction completion.
        waste_penalty = 0.0
        balance_bonus = 0.0

        total = value_pick + synergy + late_bonus + panic_penalty + block_reward + waste_penalty + balance_bonus
        signals = {
            "value_pick": round(value_pick, 4),
            "synergy": round(synergy, 4),
            "late_bonus": round(late_bonus, 4),
            "panic_penalty": round(panic_penalty, 4),
            "block_reward": round(block_reward, 4),
            "waste_penalty": round(waste_penalty, 4),
            "balance_bonus": round(balance_bonus, 4),
            "total": round(total, 4),
        }
        self.reward_signals[team_id]["value_pick"] += signals["value_pick"]
        self.reward_signals[team_id]["synergy"] += signals["synergy"]
        self.reward_signals[team_id]["late_bonus"] += signals["late_bonus"]
        self.reward_signals[team_id]["panic_penalty"] += signals["panic_penalty"]
        self.reward_signals[team_id]["block_reward"] += signals["block_reward"]
        self.reward_signals[team_id]["waste_penalty"] += signals["waste_penalty"]
        self.reward_signals[team_id]["balance_bonus"] += signals["balance_bonus"]
        self.reward_signals[team_id]["TOTAL"] += signals["total"]
        self.reward_logger.log_auction_reward(team_id, signals)
        return total

    def step(self, actions: dict) -> tuple:
        if self.done:
            return {}, self.episode_rewards, True, {"phase": "done", "message": "Episode complete"}

        assert self.auction_engine is not None
        assert self.season_simulator is not None
        assert self.transfer_market is not None

        info: dict[str, Any] = {"phase": self.phase}
        rewards_dict = {team_id: 0.0 for team_id in self.TEAMS[: self.num_teams]}
        self.phase_step += 1

        if self.phase == "auction":
            for team_id in self.TEAMS[: self.num_teams]:
                action = actions.get(team_id, ("pass", None))
                act = action[0]
                if act == "bid":
                    amount = float(action[1] if len(action) > 1 and action[1] is not None else 0.0)
                    bluff = bool(action[2]) if len(action) > 2 else False
                    self.auction_engine.submit_bid(team_id, amount, bluff=bluff)
                else:
                    self.auction_engine.pass_bid(team_id)
            if self.auction_engine.current_leader is not None and self.auction_engine.rounds_at_this_price >= 1:
                lot_result = self.auction_engine.close_lot()
                if lot_result.get("status") == "sold" and lot_result.get("winner"):
                    winner = str(lot_result["winner"])
                    player_id = int(lot_result["player_id"])
                    player = self._player_from_id(player_id)
                    squad = self._build_team_squads().get(winner, [])
                    opp_data = self.auction_engine.get_observation(winner)
                    auction_total = self.auction_reward(
                        winner, player, float(lot_result.get("price", 0.0)), squad, opp_data
                    )
                    rewards_dict[winner] += float(auction_total)
            elif self.auction_engine.current_leader is None and self.auction_engine.rounds_at_this_price >= self.num_teams:
                # No bids from any team for this lot -> mark as unsold and move on.
                self.auction_engine.close_lot()

            if self.auction_engine.is_auction_complete():
                for team_id in self.TEAMS[: self.num_teams]:
                    obs = self.auction_engine.get_observation(team_id)
                    squad = self._build_team_squads().get(team_id, [])
                    budget_left = float(obs.get("own_budget", 0.0))
                    waste_penalty = -10.0 if budget_left > 25.0 and len(squad) < 20 else 0.0
                    counts = self._role_counts(squad)
                    all_roles_filled = counts["WK"] >= 1 and counts["BAT"] >= 4 and counts["BOWL"] >= 4
                    balance_bonus = 15.0 if all_roles_filled else -5.0
                    self.reward_signals[team_id]["waste_penalty"] += waste_penalty
                    self.reward_signals[team_id]["balance_bonus"] += balance_bonus
                    self.reward_signals[team_id]["budget_wasted_cr"] = round(max(0.0, budget_left), 4)
                    self.reward_signals[team_id]["squad_balance_score"] = self._squad_balance_score(squad)
                    self.reward_signals[team_id]["TOTAL"] += waste_penalty + balance_bonus
                    rewards_dict[team_id] += waste_penalty + balance_bonus
                    self.reward_logger.log_auction_reward(
                        team_id,
                        {
                            "value_pick": 0.0,
                            "synergy": 0.0,
                            "late_bonus": 0.0,
                            "panic_penalty": 0.0,
                            "block_reward": 0.0,
                            "waste_penalty": round(waste_penalty, 4),
                            "balance_bonus": round(balance_bonus, 4),
                            "total": round(waste_penalty + balance_bonus, 4),
                        },
                    )
                self.team_squads = self._build_team_squads()
                season_teams = [{"id": tid, "squad": squad} for tid, squad in self.team_squads.items()]
                self.season_simulator = SeasonSimulator(season_teams)
                self.transfer_market = TransferMarket(
                    self._build_transfer_teams(),
                    self.team_squads,
                    self.season_simulator,
                    mid_season_point=7,
                )
                self.phase = "season"
                self.phase_step = 0
                info["phase_transition"] = "season"

        elif self.phase == "season":
            season_results = self.season_simulator.run_season()
            self.last_season_results = season_results
            for team_id in self.TEAMS[: self.num_teams]:
                season_reward = float(self.season_simulator.get_season_reward(team_id, season_results))
                self.reward_signals[team_id]["season_total"] += season_reward
                self.reward_signals[team_id]["TOTAL"] += season_reward
                final_position = float(season_results.get("standings", {}).get(team_id, {}).get("rank", 8))
                self.reward_signals[team_id]["final_position"] = final_position
                rewards_dict[team_id] += season_reward
                self.reward_logger.write(
                    episode=self.episode_idx,
                    step=self.phase_step,
                    reward=season_reward,
                    phase="season",
                    info={"team_id": team_id, "season_total": season_reward},
                )
            self.phase = "transfer"
            self.phase_step = 0
            self.transfer_market.open_window()
            info["phase_transition"] = "transfer"
            info["season"] = {"champion": season_results.get("champion")}

        elif self.phase == "transfer":
            for team_id in self.TEAMS[: self.num_teams]:
                action = actions.get(team_id, ("skip", None))
                act = action[0]
                if act == "trade" and action[1]:
                    details = action[1]
                    result = self.transfer_market.propose_trade(
                        from_team=team_id,
                        to_team=str(details.get("to_team")),
                        give_player_id=int(details.get("give_player_id")),
                        want_player_id=int(details.get("want_player_id")),
                        cash=float(details.get("cash", 0.0)),
                    )
                    info.setdefault("transfer_results", {})[team_id] = result
                else:
                    info.setdefault("transfer_results", {})[team_id] = {"accepted": False, "reason": "Skipped"}

                transfer_reward = float(self.transfer_market.get_transfer_reward(team_id))
                self.reward_signals[team_id]["transfer_total"] += transfer_reward
                self.reward_signals[team_id]["TOTAL"] += transfer_reward
                rewards_dict[team_id] += transfer_reward
                self.reward_logger.write(
                    episode=self.episode_idx,
                    step=self.phase_step,
                    reward=transfer_reward,
                    phase="transfer",
                    info={"team_id": team_id, "transfer_total": transfer_reward},
                )

            self.transfer_market.close_window()
            self.phase = "done"
            self.done = True
            info["phase_transition"] = "done"
            self.last_behavior_summary = self.behavior_detector.analyze_episode(
                self.agents,
                self.auction_engine.auction_log,
                self.last_season_results,
            )
            info["behavior_summary"] = self.last_behavior_summary
            for team_id in self.TEAMS[: self.num_teams]:
                if team_id in self.agents:
                    team_name = getattr(self.agents[team_id], "team_name", team_id)
                else:
                    team_name = team_id
                row = {
                    "episode": self.episode_idx,
                    "team_id": team_id,
                    "team_name": team_name,
                    "value_pick": round(self.reward_signals[team_id]["value_pick"], 4),
                    "synergy": round(self.reward_signals[team_id]["synergy"], 4),
                    "late_bonus": round(self.reward_signals[team_id]["late_bonus"], 4),
                    "panic_penalty": round(self.reward_signals[team_id]["panic_penalty"], 4),
                    "block_reward": round(self.reward_signals[team_id]["block_reward"], 4),
                    "waste_penalty": round(self.reward_signals[team_id]["waste_penalty"], 4),
                    "balance_bonus": round(self.reward_signals[team_id]["balance_bonus"], 4),
                    "season_total": round(self.reward_signals[team_id]["season_total"], 4),
                    "transfer_total": round(self.reward_signals[team_id]["transfer_total"], 4),
                    "TOTAL": round(self.compute_reward(team_id), 4),
                    "budget_wasted_cr": round(self.reward_signals[team_id]["budget_wasted_cr"], 4),
                    "final_position": int(self.reward_signals[team_id]["final_position"]),
                    "squad_balance_score": round(self.reward_signals[team_id]["squad_balance_score"], 4),
                }
                self.reward_logger.log_rewards_row(row)

        obs_dict = {team_id: self.get_observation(team_id) for team_id in self.TEAMS[: self.num_teams]}
        for team_id, rew in rewards_dict.items():
            self.episode_rewards[team_id] += float(rew)
        return obs_dict, rewards_dict, self.done, info

    def get_observation(self, agent_id) -> dict:
        assert self.auction_engine is not None
        assert self.transfer_market is not None

        team_id = str(agent_id)
        if team_id.isdigit():
            team_id = self.TEAMS[int(team_id)]

        phase_obs: dict[str, Any] = {}
        if self.phase == "auction":
            phase_obs = self.auction_engine.get_observation(team_id)
        elif self.phase == "season":
            phase_obs = {
                "season_ready": True,
                "fixtures_total": len(self.season_simulator.fixtures) if self.season_simulator else 0,
            }
        elif self.phase == "transfer":
            phase_obs = self.transfer_market.get_transfer_observation(team_id)
        else:
            phase_obs = {"final_rewards": self.episode_rewards.get(team_id, 0.0)}
        return {"phase": self.phase, "phase_step": self.phase_step, **phase_obs}

    def compute_reward(self, team_id) -> float:
        team_id = str(team_id)
        sig = self.reward_signals.get(team_id, {})
        return float(
            sig.get("value_pick", 0.0)
            + sig.get("synergy", 0.0)
            + sig.get("late_bonus", 0.0)
            + sig.get("panic_penalty", 0.0)
            + sig.get("block_reward", 0.0)
            + sig.get("waste_penalty", 0.0)
            + sig.get("balance_bonus", 0.0)
            + sig.get("season_total", 0.0)
            + sig.get("transfer_total", 0.0)
        )

    def render(self, mode="text") -> str:
        if mode != "text":
            return ""
        summary = {
            "phase": self.phase,
            "phase_step": self.phase_step,
            "done": self.done,
            "episode_rewards": self.episode_rewards,
        }
        return json.dumps(summary, indent=2)

    def get_info(self) -> dict:
        info = {
            "phase": self.phase,
            "done": self.done,
            "phase_step": self.phase_step,
            "episode_rewards": self.episode_rewards,
        }
        if self.last_season_results:
            info["champion"] = self.last_season_results.get("champion")
        info["behavior_summaries"] = {
            "latest": self.last_behavior_summary,
            "episodes_tracked": len(self.behavior_detector.episode_summaries),
            "story_bullets": self.behavior_detector.generate_story_bullets(),
        }
        return info


IPLEnv = IPLAuctionEnv


if __name__ == "__main__":
    env = IPLAuctionEnv()
    obs = env.reset()
    assert len(obs) == 8, f"Expected 8 obs, got {len(obs)}"
    validate(env)
    print("IPLAuctionEnv validated successfully!")
