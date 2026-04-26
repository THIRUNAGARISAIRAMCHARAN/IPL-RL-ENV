import gradio as gr
import json
import os
import pandas as pd
import plotly.graph_objects as go
from env.ipl_env import IPLAuctionEnv
from agents.base_agent import BaseIPLAgent

TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
TEAM_COLORS = {
    "MI": "#004BA0",
    "CSK": "#FFCC00",
    "RCB": "#EC1C24",
    "KKR": "#3A225D",
    "DC": "#00008B",
    "RR": "#EA1A85",
    "PBKS": "#AFBED1",
    "SRH": "#F7A721",
}
PERSONALITIES = [
    "aggressive",
    "conservative",
    "aggressive",
    "balanced",
    "role_filler",
    "conservative",
    "balanced",
    "role_filler",
]


def _empty_roster_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Player", "Role", "Price"])


def _build_team_roster_df(squads: dict, team_id: str) -> pd.DataFrame:
    players = squads.get(team_id, [])
    rows = []
    for player in players:
        rows.append(
            {
                "Player": player.get("name", "Unknown"),
                "Role": player.get("role", "N/A"),
                "Price": f"Rs.{float(player.get('price', 0.0)):.1f}Cr",
            }
        )
    return pd.DataFrame(rows, columns=["Player", "Role", "Price"]) if rows else _empty_roster_df()


def _empty_figure(title: str, msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": msg,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        template="plotly_dark",
    )
    return fig


def _build_behavior_comparison_figure(behaviors_data) -> go.Figure:
    metrics = ["overbid_rate", "block_rate", "patience_score", "bluff_success_rate"]
    metric_labels = ["Overbid", "Block", "Patience", "Bluff Success"]

    if isinstance(behaviors_data, dict):
        behaviors_data = [behaviors_data]
    if not isinstance(behaviors_data, list) or len(behaviors_data) < 20:
        return _empty_figure(
            "Before vs After Behavior",
            "Need at least 20 episodes for behavior comparison.",
        )

    early = behaviors_data[:10]
    late = behaviors_data[-10:]

    def avg_metric(episodes, metric):
        vals = [
            float(ep.get(str(i), {}).get(metric, 0.0))
            for ep in episodes
            if isinstance(ep, dict)
            for i in range(8)
        ]
        return (sum(vals) / len(vals)) if vals else 0.0

    early_vals = [avg_metric(early, m) for m in metrics]
    late_vals = [avg_metric(late, m) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Early (first 10)", x=metric_labels, y=early_vals))
    fig.add_trace(go.Bar(name="Late (last 10)", x=metric_labels, y=late_vals))
    fig.update_layout(
        title="Before vs After Behavior",
        barmode="group",
        yaxis_title="Score",
        template="plotly_dark",
    )
    return fig


def run_demo_auction():
    env = IPLAuctionEnv()
    agents = {}
    for i in range(8):
        team_id = TEAM_NAMES[i]
        agent = BaseIPLAgent(team_id, PERSONALITIES[i])
        agent.team_name = TEAM_NAMES[i]
        agents[team_id] = agent

    obs = env.reset()
    done = False
    log = []
    last_info = {}

    while not done:
        actions = {}
        for team_id in TEAM_NAMES:
            decision = agents[team_id].decide_bid(obs.get(team_id, {}))
            if decision.get("action") == "bid":
                actions[team_id] = ("bid", decision.get("amount", 0.5), decision.get("bluff", False))
            else:
                actions[team_id] = ("pass", None)

        obs, rewards, done, info = env.step(actions)
        del rewards
        last_info = info
        if "lot_closed" in info:
            lot = info["lot_closed"]
            winner = TEAM_NAMES[int(lot["winner"])] if str(lot["winner"]).isdigit() else str(lot["winner"])
            log.append(f"{lot['player_name']} -> {winner} @ Rs.{lot['price']:.1f}Cr")

    # Full Squads Table
    squads = last_info.get("final_squads", env.team_squads)
    team_roster_dfs = [_build_team_roster_df(squads, team_id) for team_id in TEAM_NAMES]

    # Season Results
    results = env.last_season_results
    standings = results.get("standings", {})
    standings_data = []
    for tid, data in standings.items():
        standings_data.append([
            data.get("rank", 8),
            tid,
            data.get("wins", 0),
            data.get("losses", 0),
            round(data.get("nrr", 0.0), 2)
        ])
    standings_df = (
        pd.DataFrame(standings_data, columns=["Rank", "Team", "Wins", "Losses", "NRR"]).sort_values("Rank")
        if standings_data
        else pd.DataFrame(columns=["Rank", "Team", "Wins", "Losses", "NRR"])
    )
    
    # All 56 Matches
    match_list = []
    for i, m in enumerate(results.get("results", []), 1):
        match_list.append([
            f"M{i}",
            f"{m['team_a']} vs {m['team_b']}",
            m['winner'],
            "Yes" if m.get("upset") else "No"
        ])
    matches_df = (
        pd.DataFrame(match_list, columns=["Match ID", "Fixture", "Winner", "Upset"])
        if match_list
        else pd.DataFrame(columns=["Match ID", "Fixture", "Winner", "Upset"])
    )

    # Playoffs & Champion
    bracket = results.get("bracket", {})
    qualification_text = "### 🏁 Playoff Qualification\n"
    top_4 = standings_df["Team"].tolist()[:4] if not standings_df.empty else []
    qualification_text += f"The top 4 teams qualified: **{', '.join(top_4)}**\n\n"
    
    bracket_text = "### 🏆 Playoff Brackets\n"
    if bracket:
        for phase in ["Q1", "Eliminator", "Q2", "Final"]:
            if phase in bracket:
                bracket_text += f"- **{phase}**: {bracket[phase]['winner']} defeated {bracket[phase]['loser']}\n"
    
    champion = results.get("champion", "N/A")
    champ_text = f"## FINAL WINNER: {champion}"

    # Transfer Activity
    transfer_results = last_info.get("transfer_results", {})
    transfer_text = "### Mid-Season Transfers\n"
    transfers_found = False
    for tid, res in transfer_results.items():
        if res.get("accepted"):
            transfers_found = True
            transfer_text += f"- **{tid}** successfully traded players.\n"
    if not transfers_found:
        transfer_text += "No trades were executed in this window."

    return (
        "\n".join(log[-30:]) if log else "No events.", 
        *team_roster_dfs,
        standings_df,
        matches_df,
        qualification_text,
        bracket_text,
        champ_text,
        transfer_text
    )


def load_training_metrics():
    reward_fig = _empty_figure("Reward per Episode", "No training logs found.")
    win_fig = _empty_figure("Win Rate vs Episodes", "No training logs found.")
    budget_fig = _empty_figure("Budget Efficiency vs Episodes", "No training logs found.")
    behavior_fig = _empty_figure("Before vs After Behavior", "No behavior logs found.")

    try:
        path = "training/logs/reward_curve.json"
        if not os.path.exists(path):
            return "No training logs found.", reward_fig, win_fig, budget_fig, behavior_fig, "No learning proof yet."
        with open(path, "r", encoding="utf-8") as f:
            curve = json.load(f)

        episodes = curve.get("episodes", [])
        teams = curve.get("teams", {})
        lines = []
        for team, metrics in teams.items():
            rewards = metrics.get("rewards", [])
            if rewards:
                avg = sum(rewards[-10:]) / max(1, len(rewards[-10:]))
                lines.append(f"{team}: Avg Reward = {avg:.1f}")
            if episodes and len(rewards) == len(episodes):
                reward_fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=rewards,
                        name=team,
                        line={"color": TEAM_COLORS.get(team, "gray")},
                    )
                )
            win_rate = metrics.get("win_rate", [])
            if episodes and len(win_rate) == len(episodes):
                win_fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=win_rate,
                        name=team,
                        line={"color": TEAM_COLORS.get(team, "gray")},
                    )
                )
            budget_eff = metrics.get("budget_efficiency", [])
            if episodes and len(budget_eff) == len(episodes):
                budget_fig.add_trace(
                    go.Scatter(
                        x=episodes,
                        y=budget_eff,
                        name=team,
                        line={"color": TEAM_COLORS.get(team, "gray")},
                    )
                )

        reward_fig.update_layout(title="Reward per Episode", xaxis_title="Episode", template="plotly_dark")
        win_fig.update_layout(title="Win Rate vs Episodes", xaxis_title="Episode", template="plotly_dark")
        budget_fig.update_layout(title="Budget Efficiency vs Episodes", xaxis_title="Episode", template="plotly_dark")

        behavior_path = "training/logs/behavior_summaries.json"
        if os.path.exists(behavior_path):
            with open(behavior_path, "r", encoding="utf-8") as f:
                behavior_data = json.load(f)
            behavior_fig = _build_behavior_comparison_figure(behavior_data)

        proof_path = "training/logs/learning_proof.json"
        proof_text = "No learning proof yet."
        if os.path.exists(proof_path):
            with open(proof_path, "r", encoding="utf-8") as f:
                proof = json.load(f)
            reward_imp = float(proof.get("reward_improvement_pct", 0.0))
            win_imp = float(proof.get("win_rate_improvement_pct", 0.0))
            proof_text = (
                f"Reward improvement: **{reward_imp:+.1f}%**  |  "
                f"Win rate improvement: **{win_imp:+.1f}%**"
            )

        return "\n".join(lines) or "No data.", reward_fig, win_fig, budget_fig, behavior_fig, proof_text
    except Exception:
        return "No training data available.", reward_fig, win_fig, budget_fig, behavior_fig, "Could not load learning proof."


with gr.Blocks(title="IPL RL Auction Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPL Multi-Agent RL Auction Environment")
    
    with gr.Tab("Phase 1: Auction"):
        with gr.Row():
            with gr.Column(scale=1):
                run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
                auction_log = gr.Textbox(label="Live Log", lines=10)
            with gr.Column(scale=2):
                gr.Markdown("### Final Rosters (Team-wise)")
                team_roster_outputs = []
                for row_start in range(0, len(TEAM_NAMES), 4):
                    with gr.Row():
                        for team_id in TEAM_NAMES[row_start: row_start + 4]:
                            with gr.Column():
                                table = gr.Dataframe(label=f"{team_id} Roster")
                                team_roster_outputs.append(table)
    
    with gr.Tab("Phase 2: Season"):
        gr.Markdown("### Simulate 56 matches, playoffs, and crown a champion")
        with gr.Row():
            standings_out = gr.Dataframe(label="Standings")
            matches_out = gr.Dataframe(label="Match Results")
        with gr.Row():
            qual_out = gr.Markdown("### Qualification")
            bracket_out = gr.Markdown("### Playoffs")
        champ_out = gr.Markdown("## WINNER: -")
        
    with gr.Tab("Phase 3: Transfer"):
        transfer_out = gr.Markdown("-")

    with gr.Tab("Training Metrics"):
        results_out = gr.Textbox(label="Reward Stats", lines=8)
        proof_out = gr.Markdown("No learning proof yet.")
        reward_plot = gr.Plot(label="Reward per Episode")
        win_plot = gr.Plot(label="Win Rate vs Episodes")
        budget_plot = gr.Plot(label="Budget Efficiency vs Episodes")
        behavior_plot = gr.Plot(label="Before vs After Behavior")
        gr.Button("Refresh Logs & Graphs").click(
            fn=load_training_metrics,
            outputs=[results_out, reward_plot, win_plot, budget_plot, behavior_plot, proof_out],
        )

    with gr.Tab("About"):
        gr.Markdown("Environment for Multi-Agent IPL Auction RL.")

    run_btn.click(
        fn=run_demo_auction, 
        outputs=[
            auction_log,
            *team_roster_outputs,
            standings_out,
            matches_out,
            qual_out,
            bracket_out,
            champ_out,
            transfer_out,
        ],
    )

if __name__ == "__main__":
    demo.launch()
