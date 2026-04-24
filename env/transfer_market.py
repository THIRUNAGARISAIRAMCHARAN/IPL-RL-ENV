from __future__ import annotations

from typing import Any


class TransferMarket:
    def __init__(self, teams, squads, season_simulator, mid_season_point=7):
        # Opens after match 7, closes before match 8
        # Max 2 trades per team
        # transfer_budget = 10% of auction budget remaining
        self.teams = teams
        self.squads = squads
        self.season_simulator = season_simulator
        self.mid_season_point = int(mid_season_point)
        self.open = False
        self.trade_log = []
        self.trade_counts = {str(self._team_id(team, idx)): 0 for idx, team in enumerate(teams)}
        self.transfer_budgets = self._build_transfer_budgets()
        self.last_trade_delta = {team_id: 0.0 for team_id in self.trade_counts}
        self.last_role_gap_fixed = {team_id: False for team_id in self.trade_counts}

    def _team_id(self, team: Any, fallback: int) -> str:
        if isinstance(team, dict):
            return str(team.get("id", fallback))
        return str(fallback)

    def _build_transfer_budgets(self) -> dict[str, float]:
        budgets = {}
        for idx, team in enumerate(self.teams):
            tid = str(self._team_id(team, idx))
            auction_budget_remaining = float(team.get("budget_cr", 90.0)) if isinstance(team, dict) else 90.0
            budgets[tid] = round(auction_budget_remaining * 0.10, 2)
        return budgets

    def _get_team_squad(self, team_id: str) -> list[dict[str, Any]]:
        raw = self.squads.get(team_id, [])
        # Normalize in case caller stores {"squad": [...]} objects.
        if isinstance(raw, dict):
            return list(raw.get("squad", []))
        return list(raw)

    def _role_counts(self, squad: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"BAT": 0, "BOWL": 0, "AR": 0, "WK": 0}
        for p in squad:
            role = p.get("role")
            if role in counts:
                counts[role] += 1
        return counts

    def _has_role_gap(self, squad: list[dict[str, Any]]) -> bool:
        counts = self._role_counts(squad)
        return counts["WK"] < 1 or counts["BAT"] < 4 or counts["BOWL"] < 4

    def _strength_contribution(self, player: dict[str, Any]) -> float:
        form = float(player.get("hidden_true_form", 1.0))
        stats = player.get("visible_stats", {})
        batting = float(stats.get("batting_avg", 0.0)) * (float(stats.get("strike_rate", 0.0)) / 100.0) * form
        bowling = (
            float(stats.get("wickets_per_match", 0.0))
            / max(0.01, float(stats.get("bowling_economy", 10.0)))
            * form
            * 20.0
        )
        return round((batting / 20.0) + bowling, 4)

    def _find_player(self, team_id: str, player_id: int) -> dict[str, Any] | None:
        for p in self._get_team_squad(team_id):
            if int(p.get("id", -1)) == int(player_id):
                return p
        return None

    def _find_player_global(self, player_id: int) -> dict[str, Any] | None:
        for team_id in self.trade_counts:
            found = self._find_player(team_id, player_id)
            if found is not None:
                return found
        return None

    def get_transfer_observation(self, team_id):
        team_id = str(team_id)
        own_squad = self._get_team_squad(team_id)
        ranked_weak = sorted(
            [
                {"id": p["id"], "role": p.get("role"), "strength_contribution": self._strength_contribution(p)}
                for p in own_squad
            ],
            key=lambda x: x["strength_contribution"],
        )

        other_comp = {}
        for tid in self.trade_counts:
            if tid == team_id:
                continue
            other_comp[tid] = self._role_counts(self._get_team_squad(tid))

        return {
            "own_weak_players": ranked_weak,
            "other_teams_role_composition": other_comp,
            "own_transfer_budget": float(self.transfer_budgets.get(team_id, 0.0)),
            "trades_remaining": 2 - int(self.trade_counts.get(team_id, 0)),
            # NOTE: other teams transfer budgets are HIDDEN
        }

    def evaluate_trade(self, team_id, give_player_id, get_player_id):
        team_id = str(team_id)
        squad = self._get_team_squad(team_id)
        give_p = self._find_player(team_id, give_player_id)
        get_p = self._find_player_global(get_player_id)

        if give_p is None or get_p is None:
            return -999.0

        old_strength = self.season_simulator.compute_team_strength(squad)
        new_squad = [p for p in squad if int(p["id"]) != int(give_player_id)]
        new_squad.append(get_p)
        new_strength = self.season_simulator.compute_team_strength(new_squad)

        delta = float(new_strength - old_strength)
        cost_factor = abs(self._strength_contribution(get_p) - self._strength_contribution(give_p))
        return delta * 10 - cost_factor * 2

    def propose_trade(self, from_team, to_team, give_player_id, want_player_id, cash=0):
        from_team = str(from_team)
        to_team = str(to_team)
        cash = float(cash)

        if not self.open:
            result = {"accepted": False, "reason": "Transfer window closed"}
            self.trade_log.append({"from_team": from_team, "to_team": to_team, "accepted": False, "reason": result["reason"]})
            return result

        if self.trade_counts.get(from_team, 0) >= 2 or self.trade_counts.get(to_team, 0) >= 2:
            result = {"accepted": False, "reason": "Trade limit exceeded"}
            self.trade_log.append({"from_team": from_team, "to_team": to_team, "accepted": False, "reason": result["reason"]})
            return result

        from_squad = self._get_team_squad(from_team)
        to_squad = self._get_team_squad(to_team)
        give_player = self._find_player(from_team, give_player_id)
        want_player = self._find_player(to_team, want_player_id)

        if give_player is None or want_player is None:
            result = {"accepted": False, "reason": "Player not found on specified team"}
            self.trade_log.append({"from_team": from_team, "to_team": to_team, "accepted": False, "reason": result["reason"]})
            return result

        if cash > self.transfer_budgets.get(from_team, 0.0):
            result = {"accepted": False, "reason": "Insufficient transfer budget"}
            self.trade_log.append({"from_team": from_team, "to_team": to_team, "accepted": False, "reason": result["reason"]})
            return result

        old_to_strength = self.season_simulator.compute_team_strength(to_squad)
        new_to_squad = [p for p in to_squad if int(p["id"]) != int(want_player_id)] + [give_player]
        new_to_strength = self.season_simulator.compute_team_strength(new_to_squad)
        to_team_strength_delta = float(new_to_strength - old_to_strength)

        # Auto-accept if strength_delta for to_team > 0.05
        accepted = to_team_strength_delta > 0.05
        reason = "Accepted: improves receiving team strength" if accepted else "Rejected: insufficient receiving team gain"

        from_old_strength = self.season_simulator.compute_team_strength(from_squad)
        from_new_squad = [p for p in from_squad if int(p["id"]) != int(give_player_id)] + [want_player]
        from_new_strength = self.season_simulator.compute_team_strength(from_new_squad)
        from_delta = float(from_new_strength - from_old_strength)

        role_gap_before = self._has_role_gap(from_squad)
        role_gap_after = self._has_role_gap(from_new_squad)

        if accepted:
            self.squads[from_team] = from_new_squad
            self.squads[to_team] = new_to_squad
            self.transfer_budgets[from_team] = round(self.transfer_budgets.get(from_team, 0.0) - cash, 2)
            self.transfer_budgets[to_team] = round(self.transfer_budgets.get(to_team, 0.0) + cash, 2)
            self.trade_counts[from_team] += 1
            self.trade_counts[to_team] += 1

        self.last_trade_delta[from_team] = from_delta if accepted else 0.0
        self.last_role_gap_fixed[from_team] = bool(role_gap_before and not role_gap_after and accepted)

        # Log all attempts (accepted AND rejected) to self.trade_log
        self.trade_log.append(
            {
                "from_team": from_team,
                "to_team": to_team,
                "give_player_id": int(give_player_id),
                "want_player_id": int(want_player_id),
                "cash": cash,
                "accepted": accepted,
                "reason": reason,
                "to_team_delta": round(to_team_strength_delta, 4),
                "from_team_delta": round(from_delta, 4),
            }
        )
        return {"accepted": accepted, "reason": reason}

    def get_transfer_reward(self, team_id):
        team_id = str(team_id)
        delta = float(self.last_trade_delta.get(team_id, 0.0))

        # +10 if strength_delta > 0.10
        # +3  if correctly skipped bad deal
        # -5  if trade lowered team strength
        # +5  if trade fixed role gap that caused match losses
        reward = 0.0
        if delta > 0.10:
            reward += 10.0
        if delta < 0.0:
            reward -= 5.0
        if self.last_role_gap_fixed.get(team_id, False):
            reward += 5.0

        # Correctly skipped bad deal: latest rejected proposal from this team where projected delta <= 0.
        for entry in reversed(self.trade_log):
            if entry.get("from_team") == team_id and not entry.get("accepted", False):
                projected = float(entry.get("from_team_delta", 0.0))
                if projected <= 0.0:
                    reward += 3.0
                break
        return float(round(reward, 4))

    def open_window(self):
        self.open = True

    def close_window(self):
        self.open = False

    # Backward-compatible wrapper used by prior env integration.
    def execute(self, team_states: dict[str, Any]) -> dict[str, Any]:
        if not self.open:
            self.open_window()
        return {"movements": [], "count": 0, "note": "Use propose_trade() for detailed transfer logic"}
