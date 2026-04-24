from __future__ import annotations

from typing import Any

from .base_agent import BaseIPLAgent


class LLMAgent(BaseIPLAgent):
    """LLM-ready wrapper with natural-language observation formatting."""

    def __init__(self, team_id: str, personality: str | None = None) -> None:
        super().__init__(team_id=team_id, personality=personality)
        self.last_prompt: str = ""

    def select_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.decide_bid(observation)

    def _observation_to_prompt(self, observation: dict[str, Any]) -> str:
        current = observation.get("current_player") or {}
        role_scarcity = observation.get("role_scarcity", {})
        return (
            f"You are IPL auction agent for {self.team_id} with personality={self.personality}. "
            f"Own budget={observation.get('own_budget', 0)}. "
            f"Current player={current.get('id')} role={current.get('role')} tier={current.get('tier')}. "
            f"Current bid={observation.get('current_bid', 0)} leader={observation.get('current_leader')}. "
            f"Active bidders={observation.get('num_active_bidders', 0)} "
            f"rounds_at_this_price={observation.get('rounds_at_this_price', 0)}. "
            f"Role scarcity={role_scarcity}. Decide bid or pass."
        )

    def decide_bid(self, observation: dict[str, Any]) -> dict[str, Any]:
        # Placeholder: create natural-language context for future LLM call.
        self.last_prompt = self._observation_to_prompt(observation)
        return super().decide_bid(observation)
