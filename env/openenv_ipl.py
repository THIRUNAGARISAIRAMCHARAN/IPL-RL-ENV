"""
Lightweight IPL auction tool environment for TRL/OpenEnv-style training.

Meta OpenEnv exposes ``Environment`` in recent releases; some installs only ship
``BaseEnvironment``. TRL only requires ``reset`` + public tool methods with
docstrings, so a minimal fallback base keeps imports working everywhere.
"""

from __future__ import annotations

import random
from typing import Any

try:
    from openenv import Environment
except ImportError:  # pragma: no cover
    try:
        from openenv import BaseEnvironment as Environment
    except ImportError:

        class Environment:  # type: ignore[too-few-public-methods]
            """Fallback when OpenEnv is not installed."""

            pass


TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]


class IPLAuctionEnvironment(Environment):
    """
    IPL Multi-Agent Auction Environment.
    8 franchises compete to build the best cricket squad within a 90Cr budget.
    Implements Theme 1 (Multi-Agent) and Theme 2 (Long-Horizon Planning).
    """

    def reset(self, **kwargs: Any) -> str:
        self.budget = 90.0
        self.squad: list[dict[str, Any]] = []
        self.round = 0
        self.done = False
        self.reward = 0.0
        self.players_remaining = 30
        self.current_player = self._generate_player()
        return (
            f"IPL Auction started!\n"
            f"League: {', '.join(TEAM_NAMES)}\n"
            f"Budget: Rs.{self.budget:.1f}Cr\n"
            f"Squad: {len(self.squad)} players\n"
            f"Current Player: {self.current_player['role']} {self.current_player['tier']}\n"
            f"Base Price: Rs.{self.current_player['base_price']:.1f}Cr\n"
            f"Players Remaining: {self.players_remaining}\n"
            f"Action: BID <amount> or PASS"
        )

    def _generate_player(self) -> dict[str, Any]:
        roles = ["BAT", "BOWL", "AR", "WK"]
        tiers = ["Tier-1", "Tier-2", "Tier-3"]
        tier = random.choice(tiers)
        base = {"Tier-1": 8.0, "Tier-2": 4.0, "Tier-3": 1.0}[tier]
        return {
            "role": random.choice(roles),
            "tier": tier,
            "base_price": base,
            "true_value": base * random.uniform(0.8, 2.0),
        }

    def bid(self, amount: float) -> str:
        """
        Place a bid on the current player.

        Args:
            amount: Bid amount in Crores (e.g. 5.5)

        Returns:
            Result of the bid with updated squad and budget info.
        """
        if self.done:
            raise ValueError("Auction is over.")

        amount = float(amount)
        if amount > self.budget:
            return f"Insufficient budget! You have Rs.{self.budget:.1f}Cr remaining."

        if amount < self.current_player["base_price"]:
            return f"Bid too low! Base price is Rs.{self.current_player['base_price']:.1f}Cr."

        # Value pick reward
        value_ratio = self.current_player["true_value"] / amount
        pick_reward = min(value_ratio - 1.0, 1.0)

        self.squad.append({**self.current_player, "price_paid": amount})
        self.budget -= amount
        self.reward += pick_reward
        self.players_remaining -= 1
        self.round += 1

        if self.players_remaining <= 0 or self.budget < 0.5:
            self.done = True
            self.reward += self._compute_squad_bonus()
            return (
                f"Won player! Squad: {len(self.squad)} | "
                f"Budget left: Rs.{self.budget:.1f}Cr | "
                f"Auction Complete! Final Reward: {self.reward:.2f}"
            )

        self.current_player = self._generate_player()
        return (
            f"Won player for Rs.{amount:.1f}Cr! "
            f"Budget left: Rs.{self.budget:.1f}Cr | "
            f"Next: {self.current_player['role']} {self.current_player['tier']} "
            f"(Base: Rs.{self.current_player['base_price']:.1f}Cr)"
        )

    def pass_player(self) -> str:
        """
        Pass on the current player and move to the next.

        Returns:
            Next player details.
        """
        if self.done:
            raise ValueError("Auction is over.")

        self.players_remaining -= 1
        self.round += 1
        self.reward -= 0.1  # small penalty for passing

        if self.players_remaining <= 0:
            self.done = True
            self.reward += self._compute_squad_bonus()
            return f"Passed. Auction Complete! Final Reward: {self.reward:.2f}"

        self.current_player = self._generate_player()
        return (
            f"Passed. Next Player: {self.current_player['role']} "
            f"{self.current_player['tier']} "
            f"(Base: Rs.{self.current_player['base_price']:.1f}Cr) | "
            f"Budget: Rs.{self.budget:.1f}Cr"
        )

    def _compute_squad_bonus(self) -> float:
        roles = [p["role"] for p in self.squad]
        bonus = 0.0
        if roles.count("BAT") >= 4:
            bonus += 1.0
        if roles.count("BOWL") >= 3:
            bonus += 1.0
        if roles.count("AR") >= 1:
            bonus += 0.5
        if roles.count("WK") >= 1:
            bonus += 0.5
        if self.budget > 10:
            bonus += 0.5  # budget efficiency
        return bonus
