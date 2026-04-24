from __future__ import annotations

import itertools
import random
from typing import Any


class SeasonSimulator:
    def __init__(self, teams_with_squads):
        # Backward compatibility for existing code path:
        # SeasonSimulator(players, rng) -> squads can be set later in run_round.
        if isinstance(teams_with_squads, list) and teams_with_squads and "role" in teams_with_squads[0]:
            self.players_by_id = {p["id"]: p for p in teams_with_squads}
            self.teams: dict[str, dict[str, Any]] = {}
            self.rng = random.Random(42)
        else:
            self.players_by_id = {}
            self.teams = self._normalize_teams(teams_with_squads)
            self.rng = random.Random(42)
            for team in self.teams.values():
                for player in team["squad"]:
                    self.players_by_id[player["id"]] = player

        # Round-robin: 8 teams x 7 opponents x 2 = 56 total matches
        self.fixtures = self._generate_fixtures()
        assert len(self.fixtures) == 56, "Fixture generation failed to produce 56 matches"
        self.last_bracket: dict[str, Any] = {}

    def _normalize_teams(self, teams_with_squads: Any) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(teams_with_squads, dict):
            iterator = teams_with_squads.items()
        else:
            iterator = enumerate(teams_with_squads or [])

        for idx, team in iterator:
            team_id = str(team.get("id", idx))
            normalized[team_id] = {
                "id": team_id,
                "squad": list(team.get("squad", [])),
            }
        return normalized

    def _generate_fixtures(self):
        team_ids = list(self.teams.keys())
        if len(team_ids) != 8:
            return []
        fixtures = []
        for a_idx, b_idx in itertools.combinations(range(len(team_ids)), 2):
            a = team_ids[a_idx]
            b = team_ids[b_idx]
            fixtures.append((a, b))
            fixtures.append((b, a))
        return fixtures

    def _player_form_after_injury(self, player: dict[str, Any]) -> float:
        form = float(player.get("hidden_true_form", 1.0))
        injury_risk = float(player.get("injury_risk", 0.0))
        # injury_draw = random.random() < injury_risk -> apply 0.7x form
        if self.rng.random() < injury_risk:
            form *= 0.7
        return max(0.4, form)

    def compute_team_strength(self, squad, pitch_type="neutral"):
        if not squad:
            return 0.5

        batting_scores = []
        bowling_scores = []
        role_counts = {"WK": 0, "BAT": 0, "BOWL": 0}
        synergy_tags_per_player = []

        for player in squad:
            stats = player.get("visible_stats", {})
            role = player.get("role")
            if role in role_counts:
                role_counts[role] += 1

            form = self._player_form_after_injury(player)
            batting_score = float(stats.get("batting_avg", 0.0)) * (float(stats.get("strike_rate", 0.0)) / 100.0) * form
            bowling_score = (float(stats.get("wickets_per_match", 0.0)) / max(0.01, float(stats.get("bowling_economy", 10.0)))) * form
            batting_scores.append((batting_score, player))
            bowling_scores.append((bowling_score, player))
            synergy_tags_per_player.append(set(player.get("synergy_tags", [])))

        # batting_strength = avg(top-6 batters: batting_avg * sr/100 * hidden_true_form)
        top_bat = sorted(batting_scores, key=lambda x: x[0], reverse=True)[:6]
        batting_strength_raw = sum(s for s, _ in top_bat) / max(1, len(top_bat))

        # bowling_strength = avg(top-4 bowlers: wkts_per_match / economy * hidden_true_form)
        top_bowl = sorted(bowling_scores, key=lambda x: x[0], reverse=True)[:4]
        bowling_strength_raw = sum(s for s, _ in top_bowl) / max(1, len(top_bowl))

        # Normalization to keep aggregate strength in target range.
        batting_component = batting_strength_raw / 55.0
        bowling_component = bowling_strength_raw * 5.0

        # synergy_bonus = +0.05 per pair of players sharing a synergy_tag
        shared_pairs = 0
        for i in range(len(synergy_tags_per_player)):
            for j in range(i + 1, len(synergy_tags_per_player)):
                if synergy_tags_per_player[i].intersection(synergy_tags_per_player[j]):
                    shared_pairs += 1
        synergy_bonus = shared_pairs * 0.05

        # balance_bonus = +0.10 if squad has >=1 WK, >=4 BAT, >=4 BOWL
        balance_bonus = 0.10 if role_counts["WK"] >= 1 and role_counts["BAT"] >= 4 and role_counts["BOWL"] >= 4 else 0.0

        # depth_penalty = -0.05 per slot below 15 players
        depth_penalty = max(0, 15 - len(squad)) * 0.05

        pitch_mod = {"neutral": 1.0, "spin": 1.02, "pace": 1.02}.get(pitch_type, 1.0)
        strength = (0.55 * batting_component + 0.45 * bowling_component) * pitch_mod
        strength = strength + synergy_bonus + balance_bonus - depth_penalty

        # return strength float in 0.5-2.0 range
        return max(0.5, min(2.0, round(strength, 4)))

    def simulate_match(self, team_a_id, team_b_id):
        team_a = self.teams[str(team_a_id)]
        team_b = self.teams[str(team_b_id)]

        base_a = self.compute_team_strength(team_a["squad"])
        base_b = self.compute_team_strength(team_b["squad"])

        # Match-day noise: multiply each strength by random.uniform(0.88, 1.12)
        sa = round(base_a * self.rng.uniform(0.88, 1.12), 4)
        sb = round(base_b * self.rng.uniform(0.88, 1.12), 4)

        winner = str(team_a_id) if sa >= sb else str(team_b_id)
        loser = str(team_b_id) if winner == str(team_a_id) else str(team_a_id)

        # Track upsets: lower-strength team wins
        weaker_team = str(team_a_id) if base_a < base_b else str(team_b_id)
        upset = winner == weaker_team and base_a != base_b

        return {"winner": winner, "loser": loser, "sa": sa, "sb": sb, "upset": upset}

    def run_season(self):
        standings: dict[str, dict[str, Any]] = {
            tid: {"wins": 0, "losses": 0, "nrr": 0.0} for tid in self.teams
        }
        results = []

        # Run all 56 fixtures, update standings
        for team_a, team_b in self.fixtures:
            result = self.simulate_match(team_a, team_b)
            winner = result["winner"]
            loser = result["loser"]
            standings[winner]["wins"] += 1
            standings[loser]["losses"] += 1

            if winner == team_a:
                standings[team_a]["nrr"] += result["sa"] - result["sb"]
                standings[team_b]["nrr"] += result["sb"] - result["sa"]
            else:
                standings[team_a]["nrr"] += result["sa"] - result["sb"]
                standings[team_b]["nrr"] += result["sb"] - result["sa"]
            results.append({"team_a": team_a, "team_b": team_b, **result})

        ranked = sorted(standings.items(), key=lambda x: (x[1]["wins"], x[1]["nrr"]), reverse=True)
        for rank, (tid, row) in enumerate(ranked, start=1):
            standings[tid]["rank"] = rank

        # top_4 = sort by wins then nrr
        top_4 = [tid for tid, _ in ranked[:4]]
        champion = self.run_playoffs(top_4)
        return {"standings": standings, "champion": champion, "results": results, "bracket": self.last_bracket}

    def run_playoffs(self, top_4):
        # Q1: 1st vs 2nd | Eliminator: 3rd vs 4th
        q1 = self.simulate_match(top_4[0], top_4[1])
        eliminator = self.simulate_match(top_4[2], top_4[3])

        # Q2: loser(Q1) vs winner(Elim) | Final: winner(Q1) vs winner(Q2)
        q2 = self.simulate_match(q1["loser"], eliminator["winner"])
        final = self.simulate_match(q1["winner"], q2["winner"])
        champion_team_id = final["winner"]

        self.last_bracket = {
            "Q1": q1,
            "Eliminator": eliminator,
            "Q2": q2,
            "Final": final,
        }
        return champion_team_id

    def get_season_reward(self, team_id, results):
        team_key = str(team_id)
        standings = results.get("standings", {})
        champion = str(results.get("champion"))
        row = standings.get(team_key, {})

        wins = int(row.get("wins", 0))
        rank = int(row.get("rank", 8))
        reward = 15.0 * wins

        upset_wins = 0
        for r in results.get("results", []):
            if r.get("upset") and r.get("winner") == team_key:
                upset_wins += 1
        reward += 25.0 * upset_wins

        if rank <= 4:
            reward += 30.0
        if rank <= 2:
            reward += 50.0
        if team_key == champion:
            reward += 100.0
        if rank in (7, 8):
            reward -= 20.0
        return float(round(reward, 4))
