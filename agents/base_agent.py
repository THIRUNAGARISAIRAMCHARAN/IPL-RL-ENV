from __future__ import annotations

import random
from typing import Any


class BaseIPLAgent:
    PERSONALITIES = {
        "aggressive": {"overbid_factor": 1.25, "bluff_prob": 0.15, "role_urgency_weight": 0.5},
        "conservative": {"overbid_factor": 0.80, "bluff_prob": 0.02, "role_urgency_weight": 1.0},
        "balanced": {"overbid_factor": 1.05, "bluff_prob": 0.08, "role_urgency_weight": 0.75},
        "role_filler": {"overbid_factor": 1.40, "bluff_prob": 0.05, "role_urgency_weight": 2.0},
    }

    TEAM_PERSONALITY = {
        "MI": "aggressive",
        "RCB": "aggressive",
        "CSK": "conservative",
        "RR": "conservative",
        "KKR": "balanced",
        "PBKS": "balanced",
        "DC": "role_filler",
        "SRH": "role_filler",
    }

    ROLE_TARGETS = {"BAT": 5, "BOWL": 5, "AR": 3, "WK": 2}

    def __init__(self, team_id: str, personality: str | None = None, seed: int = 42) -> None:
        self.team_id = str(team_id)
        self.rng = random.Random(seed)
        self.personality = personality or self.TEAM_PERSONALITY.get(self.team_id, "balanced")
        self.params = self.PERSONALITIES.get(self.personality, self.PERSONALITIES["balanced"])

        self.decision_count = 0
        self.overbid_count = 0
        self.block_attempt_count = 0
        self.bluff_attempt_count = 0
        self.bluff_success_count = 0
        self.pass_streak = 0
        self.total_pass_streak = 0
        self.finalized_lots = 0

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.decide_bid(observation)

    def _squad_role_counts(self, own_squad: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
        for p in own_squad:
            role = p.get("role")
            if role in counts:
                counts[role] += 1
        return counts

    def value_player(self, player_obs, own_squad):
        if not player_obs:
            return 0.0

        role = player_obs.get("role", "BAT")
        stats = player_obs.get("visible_stats", {})
        counts = self._squad_role_counts(own_squad)

        batting_signal = float(stats.get("batting_avg", 0.0)) * 0.02 + float(stats.get("strike_rate", 0.0)) * 0.006
        bowling_signal = max(0.0, 12.0 - float(stats.get("bowling_economy", 12.0))) * 0.2
        wickets_signal = float(stats.get("wickets_per_match", 0.0)) * 1.5
        tier_bonus = {"MARQUEE": 1.0, "A": 0.7, "B": 0.4, "C": 0.2}.get(player_obs.get("tier", "C"), 0.2)

        base_value = batting_signal + bowling_signal + wickets_signal + tier_bonus
        role_gap = max(0, self.ROLE_TARGETS.get(role, 0) - counts.get(role, 0))
        role_urgency = role_gap * float(self.params["role_urgency_weight"])
        return round(base_value + role_urgency, 4)

    def opponent_model(self, opponent_data, current_player):
        if not current_player:
            return 0.0
        role = current_player.get("role")
        pressure = 0.0
        for team_id, gaps in opponent_data.get("opponent_role_gaps", {}).items():
            del team_id
            if gaps.get(role):
                pressure += 0.12
        return round(min(1.0, pressure), 4)

    def should_block(self, obs):
        current_player = obs.get("current_player")
        if not current_player:
            return False

        opponent_need = self.opponent_model(obs, current_player)
        current_bid = float(obs.get("current_bid", 0.0))
        own_budget = float(obs.get("own_budget", 0.0))
        scarcity = obs.get("role_scarcity", {}).get(current_player.get("role"), 0)

        low_scarcity = scarcity <= 6
        cheap_enough = current_bid <= max(0.8, own_budget * 0.08)
        return bool(opponent_need > 0.45 and low_scarcity and cheap_enough)

    def decide_bid(self, observation):
        self.decision_count += 1
        current_player = observation.get("current_player")
        if not current_player:
            self.pass_streak += 1
            self.total_pass_streak += 1
            return {"team_id": self.team_id, "action": "pass"}

        own_budget = float(observation.get("own_budget", 0.0))
        own_squad = observation.get("own_squad", [])
        current_bid = float(observation.get("current_bid", 0.0))
        scarcity = observation.get("role_scarcity", {}).get(current_player.get("role"), 1)
        scarcity_boost = 1.0 + (0.25 if scarcity <= 8 else 0.0)

        valuation = self.value_player(current_player, own_squad)
        target_price = max(current_bid + 0.05, valuation * self.params["overbid_factor"] * scarcity_boost)
        block = self.should_block(observation)
        if block:
            self.block_attempt_count += 1
            target_price = max(target_price, current_bid + 0.15)

        affordable_limit = max(0.0, own_budget * 0.22)
        if target_price > affordable_limit or own_budget <= 0.25:
            self.pass_streak += 1
            self.total_pass_streak += 1
            return {"team_id": self.team_id, "action": "pass"}

        self.pass_streak = 0
        overbid_threshold = max(0.0, current_bid) * 1.1
        if target_price > overbid_threshold:
            self.overbid_count += 1

        bluff = self.rng.random() < float(self.params["bluff_prob"])
        if bluff:
            self.bluff_attempt_count += 1

        return {
            "team_id": self.team_id,
            "action": "bid",
            "amount": round(target_price, 2),
            "bluff": bluff,
            "block": block,
        }

    def record_bluff_result(self, success: bool) -> None:
        if success:
            self.bluff_success_count += 1

    def get_behavior_summary(self):
        decisions = max(1, self.decision_count)
        bluff_attempts = max(1, self.bluff_attempt_count)
        patience_score = self.total_pass_streak / decisions * 10.0
        return {
            "overbid_rate": round(self.overbid_count / decisions, 4),
            "block_rate": round(self.block_attempt_count / decisions, 4),
            "bluff_success_rate": round(self.bluff_success_count / bluff_attempts, 4),
            "patience_score": round(patience_score, 4),
        }

    def _classify_strategy(self, summary):
        if summary["overbid_rate"] > 0.4:
            return "Reckless Bidder"
        if summary["block_rate"] > 0.2:
            return "Strategic Blocker"
        if summary["patience_score"] > 3:
            return "Patient Planner"
        if summary["bluff_success_rate"] > 0.5:
            return "Deceptive Dealer"
        return "Balanced Bidder"
