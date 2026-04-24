from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


class AuctionEngine:
    """Auction phase simulator with partial observability and bluff signals."""

    ROLE_MIN_TARGETS = {"BAT": 3, "BOWL": 3, "AR": 2, "WK": 1}

    def __init__(self, players, teams, num_teams=8):
        # Backward compatibility: allow callers to pass rng in place of num_teams.
        if isinstance(num_teams, random.Random):
            self.rng = num_teams
            self.num_teams = len(teams)
        else:
            self.rng = random.Random(42)
            self.num_teams = int(num_teams)

        self.players = players
        self.teams = teams[: self.num_teams]
        self.players_by_id = {p["id"]: p for p in players}

        self.team_states: dict[str, dict[str, Any]] = {}
        self.player_queue: list[int] = []
        self.current_player_id: int | None = None
        self.current_bid: float = 0.0
        self.current_leader: str | None = None
        self.rounds_at_this_price: int = 0
        self.active_bidders: set[str] = set()

        # Required trackers.
        self.bids_per_player: dict[int, list[dict[str, Any]]] = {}
        self.unsold_pool: list[int] = []
        self.bluff_history: dict[str, list[dict[str, Any]]] = {}
        self.auction_log: list[dict[str, Any]] = []
        self.flags: dict[str, dict[str, bool]] = {}

        self._initialize_team_states()
        self._build_player_pool()
        self._start_next_lot()

    def _initialize_team_states(self) -> None:
        self.team_states = {}
        for idx, team in enumerate(self.teams):
            team_id = str(team.get("id", idx))
            self.team_states[team_id] = {
                "budget": float(team.get("budget_cr", 90)),
                "squad": [],
                "max_squad": 25,
                "min_squad": 11,
            }
            self.bluff_history[team_id] = []
            self.flags[team_id] = {"DESPERATION": False}

    def _build_player_pool(self) -> None:
        marquee = [p["id"] for p in self.players if p.get("tier") == "MARQUEE"]
        tier_a = [p["id"] for p in self.players if p.get("tier") == "A"]
        tier_b = [p["id"] for p in self.players if p.get("tier") == "B"]
        tier_c = [p["id"] for p in self.players if p.get("tier") == "C"]
        self.rng.shuffle(tier_a)
        self.rng.shuffle(tier_b)
        self.rng.shuffle(tier_c)
        # MARQUEE first, then shuffled by tier.
        self.player_queue = marquee + tier_a + tier_b + tier_c

    def _start_next_lot(self) -> None:
        self.current_bid = 0.0
        self.current_leader = None
        self.rounds_at_this_price = 0
        self.active_bidders = set(self.team_states.keys())
        if self.player_queue:
            self.current_player_id = self.player_queue[0]
            self.bids_per_player.setdefault(self.current_player_id, [])
        else:
            self.current_player_id = None

    def _resolve_team_id(self, team_id: Any) -> str:
        key = str(team_id)
        if key in self.team_states:
            return key
        if isinstance(team_id, int):
            team_ids = list(self.team_states.keys())
            if 0 <= team_id < len(team_ids):
                return team_ids[team_id]
        raise KeyError(f"Unknown team_id: {team_id}")

    def _role_counts_for_team(self, team_id: str) -> dict[str, int]:
        counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
        for pid in self.team_states[team_id]["squad"]:
            role = self.players_by_id[pid]["role"]
            counts[role] += 1
        return counts

    def _role_gap_flags(self, team_id: str) -> dict[str, bool]:
        counts = self._role_counts_for_team(team_id)
        return {role: counts[role] < minimum for role, minimum in self.ROLE_MIN_TARGETS.items()}

    def remaining_players_count(self) -> int:
        return len(self.player_queue)

    def get_observation(self, team_id):
        # CRITICAL: NEVER expose hidden_true_form or injury_risk.
        resolved_team = self._resolve_team_id(team_id)
        own_state = self.team_states[resolved_team]

        opponent_budgets = {}
        opponent_squad_sizes = {}
        opponent_role_gaps = {}
        for other_team_id, state in self.team_states.items():
            if other_team_id == resolved_team:
                continue
            noisy_multiplier = 1.0 + float(np.random.normal(0, 0.2))
            noisy_budget = max(0.0, state["budget"] * noisy_multiplier)
            opponent_budgets[other_team_id] = round(noisy_budget, 2)
            opponent_squad_sizes[other_team_id] = len(state["squad"])
            opponent_role_gaps[other_team_id] = self._role_gap_flags(other_team_id)

        current_player = None
        if self.current_player_id is not None:
            p = self.players_by_id[self.current_player_id]
            current_player = {
                "id": p["id"],
                "role": p["role"],
                "tier": p["tier"],
                "nationality": p["nationality"],
                "visible_stats": p["visible_stats"],
            }

        role_scarcity = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
        for pid in self.player_queue:
            role_scarcity[self.players_by_id[pid]["role"]] += 1

        players_needed = max(0, own_state["min_squad"] - len(own_state["squad"]))
        budget_pressure = own_state["budget"] / max(1.0, players_needed + 1.0)

        own_squad = []
        for pid in own_state["squad"]:
            player = self.players_by_id[pid]
            own_squad.append({"id": pid, "role": player["role"]})

        return {
            "own_budget": float(round(own_state["budget"], 2)),
            "own_squad": own_squad,
            "opponent_budgets": opponent_budgets,
            "opponent_squad_sizes": opponent_squad_sizes,
            "opponent_role_gaps": opponent_role_gaps,
            "current_player": current_player,
            "current_bid": float(round(self.current_bid, 2)),
            "current_leader": self.current_leader,
            "num_active_bidders": len(self.active_bidders),
            "rounds_at_this_price": self.rounds_at_this_price,
            "players_remaining": len(self.player_queue),
            "role_scarcity": role_scarcity,
            "budget_pressure": float(round(budget_pressure, 4)),
        }

    def submit_bid(self, team_id, amount, bluff=False):
        if self.current_player_id is None:
            return {"ok": False, "reason": "Auction already complete"}

        try:
            resolved_team = self._resolve_team_id(team_id)
        except KeyError as exc:
            return {"ok": False, "reason": str(exc)}

        state = self.team_states[resolved_team]
        amount = float(amount)
        if len(state["squad"]) >= state["max_squad"]:
            return {"ok": False, "reason": "Squad full"}
        if amount <= self.current_bid:
            return {"ok": False, "reason": "Bid must exceed current_bid"}
        if amount > state["budget"]:
            return {"ok": False, "reason": "Insufficient budget"}

        player = self.players_by_id[self.current_player_id]
        if amount < float(player["base_price_cr"]):
            return {"ok": False, "reason": "Bid below base price"}

        panic = self.check_panic_conditions(player)
        final_amount = amount
        if panic["panic_noise_applied"]:
            final_amount = amount * (1 + self.rng.uniform(-0.08, 0.08))
            final_amount = round(max(self.current_bid + 0.01, final_amount), 2)
            if final_amount > state["budget"]:
                final_amount = round(state["budget"], 2)

        bluff_signal = final_amount
        if bluff:
            bluff_signal = round(final_amount * 1.15, 2)
            self.bluff_history[resolved_team].append(
                {
                    "player_id": self.current_player_id,
                    "actual_bid": round(final_amount, 2),
                    "signal_bid": bluff_signal,
                }
            )

        self.current_bid = round(final_amount, 2)
        self.current_leader = resolved_team
        self.active_bidders.add(resolved_team)
        self.rounds_at_this_price = 0

        bid_event = {
            "team_id": resolved_team,
            "amount": self.current_bid,
            "bluff": bool(bluff),
            "signal_amount": bluff_signal,
            "panic": panic,
        }
        self.bids_per_player[self.current_player_id].append(bid_event)
        self.auction_log.append({"event": "bid", "player_id": self.current_player_id, **bid_event})
        return {"ok": True, "bid": self.current_bid, "leader": self.current_leader, "panic": panic}

    def check_panic_conditions(self, player):
        panic_noise_applied = player.get("tier") == "MARQUEE" and len(self.active_bidders) >= 4
        for team_id, state in self.team_states.items():
            needs_players = state["max_squad"] - len(state["squad"])
            desperation = state["budget"] < 10 and needs_players >= 5
            self.flags[team_id]["DESPERATION"] = desperation
        return {
            "panic_noise_applied": panic_noise_applied,
            "desperation_flags": {tid: vals["DESPERATION"] for tid, vals in self.flags.items()},
        }

    def pass_bid(self, team_id):
        try:
            resolved_team = self._resolve_team_id(team_id)
        except KeyError as exc:
            return {"ok": False, "reason": str(exc)}

        self.active_bidders.discard(resolved_team)
        self.rounds_at_this_price += 1
        self.auction_log.append(
            {
                "event": "pass",
                "team_id": resolved_team,
                "player_id": self.current_player_id,
                "current_leader": self.current_leader,
                "rounds_at_this_price": self.rounds_at_this_price,
            }
        )

        # If no competition remains after a leader exists, close lot quickly.
        if self.current_leader and len(self.active_bidders) <= 1:
            return self.close_lot()
        return {"ok": True, "active_bidders": len(self.active_bidders)}

    def close_lot(self):
        if not self.player_queue:
            return {"ok": False, "reason": "No players remaining"}
        if self.current_player_id is None:
            return {"ok": False, "reason": "No active lot"}

        player_id = self.current_player_id
        player = self.players_by_id[player_id]
        winner = self.current_leader

        if winner is None:
            self.unsold_pool.append(player_id)
            self.auction_log.append({"event": "unsold", "player_id": player_id})
            self.player_queue.pop(0)
            self._start_next_lot()
            return {"ok": True, "status": "unsold", "player_id": player_id}

        state = self.team_states[winner]
        final_price = round(self.current_bid, 2)
        if final_price > state["budget"]:
            # Safety fallback.
            self.unsold_pool.append(player_id)
            self.auction_log.append(
                {
                    "event": "unsold_budget_failure",
                    "player_id": player_id,
                    "winner": winner,
                    "attempted_price": final_price,
                }
            )
        else:
            state["budget"] = round(state["budget"] - final_price, 2)
            state["squad"].append(player_id)
            self.auction_log.append(
                {
                    "event": "sold",
                    "team_id": winner,
                    "player_id": player_id,
                    "price": final_price,
                    "tier": player["tier"],
                    "role": player["role"],
                }
            )

        self.player_queue.pop(0)
        self._start_next_lot()
        return {
            "ok": True,
            "status": "sold" if winner else "unsold",
            "player_id": player_id,
            "winner": winner,
            "price": final_price,
        }

    def is_auction_complete(self):
        return self.current_player_id is None and not self.player_queue


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    players = json.loads((data_dir / "players.json").read_text(encoding="utf-8"))
    teams = json.loads((data_dir / "teams.json").read_text(encoding="utf-8"))
    engine = AuctionEngine(players, teams)
    obs = engine.get_observation(0)
    assert "hidden_true_form" not in str(obs), "FAIL: hidden field exposed!"
    assert "injury_risk" not in str(obs), "FAIL: hidden field exposed!"
    print("AuctionEngine OK - hidden fields secure")
