import gradio as gr
import json
import os
import pandas as pd
import plotly.graph_objects as go
from env.ipl_env import IPLAuctionEnv
from agents.base_agent import BaseIPLAgent
import threading
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _start_training():
    try:
        print("===== Training Thread Started =====", flush=True)
        from training.train import run_training
        run_training(episodes=100)
        print("===== Training Complete =====", flush=True)
    except Exception as e:
        import traceback
        print(f"Training error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)


_training_thread = threading.Thread(target=_start_training, daemon=True)
_training_thread.start()

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
    return pd.DataFrame(columns=["Player", "Role", "Price", "Team Spent (Cr)"])


def _build_team_roster_df(squads: dict, team_id: str) -> pd.DataFrame:
    players = squads.get(team_id, [])
    total_spent = sum(float(player.get("price", 0.0)) for player in players)
    rows = []
    for player in players:
        rows.append(
            {
                "Player": player.get("name", "Unknown"),
                "Role": player.get("role", "N/A"),
                "Price": f"Rs.{float(player.get('price', 0.0)):.1f}Cr",
                "Team Spent (Cr)": round(total_spent, 2),
            }
        )
    return (
        pd.DataFrame(rows, columns=["Player", "Role", "Price", "Team Spent (Cr)"])
        if rows
        else _empty_roster_df()
    )


def _team_label(raw_team_id: str) -> str:
    raw = str(raw_team_id)
    if raw in TEAM_NAMES:
        return raw
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(TEAM_NAMES):
            return TEAM_NAMES[idx]
    return raw


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


def _build_notebook_graphs() -> tuple[str, go.Figure, go.Figure, pd.DataFrame]:
    reward_fig = _empty_figure("Reward per Episode", "No reward data yet.")
    win_fig = _empty_figure("Win Rate vs Episodes", "No reward data yet.")

    csv_path = "training/logs/rewards.csv"
    if not os.path.exists(csv_path):
        graph_table = pd.DataFrame(
            [
                {"Graph": "Reward per Episode", "Metric": "TOTAL reward", "Source": "training/logs/rewards.csv"},
                {"Graph": "Win Rate vs Episodes", "Metric": "Top-4 finish rolling win rate", "Source": "training/logs/rewards.csv"},
            ]
        )
        return "No training logs found.", reward_fig, win_fig, graph_table

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        graph_table = pd.DataFrame(
            [
                {"Graph": "Reward per Episode", "Metric": "TOTAL reward", "Source": "training/logs/rewards.csv"},
                {"Graph": "Win Rate vs Episodes", "Metric": "Top-4 finish rolling win rate", "Source": "training/logs/rewards.csv"},
            ]
        )
        return "Could not read training logs.", reward_fig, win_fig, graph_table

    if df.empty or "team_id" not in df.columns:
        graph_table = pd.DataFrame(
            [
                {"Graph": "Reward per Episode", "Metric": "TOTAL reward", "Source": "training/logs/rewards.csv"},
                {"Graph": "Win Rate vs Episodes", "Metric": "Top-4 finish rolling win rate", "Source": "training/logs/rewards.csv"},
            ]
        )
        return "No training data available.", reward_fig, win_fig, graph_table

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").fillna(0).astype(int)
    df["TOTAL"] = pd.to_numeric(df.get("TOTAL", 0), errors="coerce").fillna(0.0)
    df["final_position"] = pd.to_numeric(df.get("final_position", 8), errors="coerce").fillna(8.0)
    df["is_top4"] = (df["final_position"] <= 4).astype(float)

    # Use the latest 100 episodes so graphs reflect current runs.
    if len(df) > 0:
        latest_episode = int(df["episode"].max())
        min_episode = max(0, latest_episode - 99)
        df = df[df["episode"] >= min_episode].copy()

    # Match Colab-style line graphs.
    reward_fig = go.Figure()
    win_fig = go.Figure()
    lines = []
    for team in TEAM_NAMES:
        tdf = df[df["team_id"] == team].sort_values("episode")
        if tdf.empty:
            continue
        eps = tdf["episode"].tolist()
        rewards = tdf["TOTAL"].astype(float).tolist()
        win_rate = (
            (tdf["final_position"].astype(float) <= 4)
            .astype(float)
            .rolling(10, min_periods=1)
            .mean()
            .tolist()
        )
        reward_fig.add_trace(
            go.Scatter(x=eps, y=rewards, mode="lines", name=team, line={"color": TEAM_COLORS.get(team, "gray")})
        )
        win_fig.add_trace(
            go.Scatter(x=eps, y=win_rate, mode="lines", name=team, line={"color": TEAM_COLORS.get(team, "gray")})
        )
        lines.append(f"{team}: Avg Reward = {float(tdf['TOTAL'].tail(10).mean()):.1f}")

    reward_fig.update_layout(
        title="Reward per Episode (Latest 100)",
        xaxis_title="Episode",
        yaxis_title="Reward",
        template="plotly_dark",
    )
    win_fig.update_layout(
        title="Win Rate vs Episodes (Latest 100)",
        xaxis_title="Episode",
        yaxis_title="Win Rate",
        template="plotly_dark",
    )
    graph_table = pd.DataFrame(
        [
            {"Graph": "Reward per Episode", "Metric": "TOTAL reward", "Source": "training/logs/rewards.csv"},
            {"Graph": "Win Rate vs Episodes", "Metric": "Top-4 finish rolling win rate", "Source": "training/logs/rewards.csv"},
        ]
    )
    return "\n".join(lines) or "No data.", reward_fig, win_fig, graph_table


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
            _team_label(tid),
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
        team_a = _team_label(m.get("team_a", ""))
        team_b = _team_label(m.get("team_b", ""))
        winner = _team_label(m.get("winner", ""))
        toss_winner = _team_label(m.get("toss_winner", ""))
        toss_decision = str(m.get("toss_decision", "-"))
        xi_a = ", ".join(m.get("team_a_xi", [])[:11])
        xi_b = ", ".join(m.get("team_b_xi", [])[:11])
        match_list.append([
            f"M{i}",
            f"{team_a} vs {team_b}",
            winner,
            toss_winner,
            toss_decision,
            "Yes" if m.get("upset") else "No"
            ,
            xi_a,
            xi_b,
        ])
    matches_df = (
        pd.DataFrame(
            match_list,
            columns=[
                "Match ID",
                "Fixture",
                "Winner",
                "Toss Winner",
                "Toss Decision",
                "Upset",
                "Team A Playing XI",
                "Team B Playing XI",
            ],
        )
        if match_list
        else pd.DataFrame(
            columns=[
                "Match ID",
                "Fixture",
                "Winner",
                "Toss Winner",
                "Toss Decision",
                "Upset",
                "Team A Playing XI",
                "Team B Playing XI",
            ]
        )
    )

    # Playoff matches table (Q1, Eliminator, Q2, Final)
    bracket = results.get("bracket", {})
    playoffs_rows = []
    for stage in ["Q1", "Eliminator", "Q2", "Final"]:
        stage_data = bracket.get(stage, {})
        if not isinstance(stage_data, dict):
            continue
        p_team_a = _team_label(stage_data.get("team_a", ""))
        p_team_b = _team_label(stage_data.get("team_b", ""))
        p_winner = _team_label(stage_data.get("winner", ""))
        p_toss_winner = _team_label(stage_data.get("toss_winner", ""))
        p_toss_decision = str(stage_data.get("toss_decision", "-"))
        playoffs_rows.append(
            [
                stage,
                f"{p_team_a} vs {p_team_b}",
                p_winner,
                p_toss_winner,
                p_toss_decision,
            ]
        )
    playoffs_df = (
        pd.DataFrame(
            playoffs_rows,
            columns=["Playoff Match", "Fixture", "Winner", "Toss Winner", "Toss Decision"],
        )
        if playoffs_rows
        else pd.DataFrame(columns=["Playoff Match", "Fixture", "Winner", "Toss Winner", "Toss Decision"])
    )

    # Playoffs & Champion
    qualification_text = "### 🏁 Playoff Qualification\n"
    top_4 = standings_df["Team"].tolist()[:4] if not standings_df.empty else []
    qualification_text += f"The top 4 teams qualified: **{', '.join(top_4)}**\n\n"
    
    bracket_text = "### 🏆 Playoff Brackets\n"
    if bracket:
        for phase in ["Q1", "Eliminator", "Q2", "Final"]:
            if phase in bracket:
                w = _team_label(bracket[phase].get("winner", ""))
                l = _team_label(bracket[phase].get("loser", ""))
                bracket_text += f"- **{phase}**: {w} defeated {l}\n"
    
    champion = results.get("champion", "N/A")
    champ_text = f"## FINAL WINNER: {_team_label(champion)}"

    metrics_stats, metrics_fig_1, metrics_fig_2, metrics_table = _build_notebook_graphs()

    return (
        *team_roster_dfs,
        standings_df,
        matches_df,
        playoffs_df,
        qualification_text,
        bracket_text,
        champ_text,
        metrics_stats,
        metrics_fig_1,
        metrics_fig_2,
        metrics_table,
    )


def load_training_metrics():
    return _build_notebook_graphs()


with gr.Blocks(title="IPL RL Auction Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPL Multi-Agent RL Auction Environment")
    
    with gr.Tab("Phase 1: Auction"):
        with gr.Row():
            with gr.Column(scale=1):
                run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
                gr.Markdown(
                    "Run the full cycle to populate team-wise final rosters, season table, "
                    "all 56 match results and playoffs."
                )
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
        gr.Markdown("### Season Results: 56 league matches + playoffs")
        with gr.Row():
            standings_out = gr.Dataframe(label="Standings")
        matches_out = gr.Dataframe(label="All 56 Match Results + Playing XI")
        playoffs_out = gr.Dataframe(label="Playoffs (4 Matches) with Toss Details")
        with gr.Row():
            qual_out = gr.Markdown("### Qualification")
            bracket_out = gr.Markdown("### Playoffs")
        champ_out = gr.Markdown("## WINNER: -")
        
    with gr.Tab("All Graphs"):
        results_out = gr.Textbox(label="Performance Summary", lines=4)
        reward_plot = gr.Plot(label="Reward per Episode")
        win_plot = gr.Plot(label="Win Rate vs Episodes")
        graph_table_out = gr.Dataframe(label="Graph Data Table")
        gr.Button("Refresh Logs & Graphs").click(
            fn=load_training_metrics,
            outputs=[results_out, reward_plot, win_plot, graph_table_out],
        )

    with gr.Tab("About"):
        gr.Markdown(
            """
### Results View

- Phase 1: Final team rosters with spend.
- Phase 2: Full league and playoff results with toss outcomes.
- All Graphs: reward and win-rate graphs with data table.
"""
        )

    run_btn.click(
        fn=run_demo_auction, 
        outputs=[
            *team_roster_outputs,
            standings_out,
            matches_out,
            playoffs_out,
            qual_out,
            bracket_out,
            champ_out,
            results_out,
            reward_plot,
            win_plot,
            graph_table_out,
        ],
    )
    demo.load(
        fn=load_training_metrics,
        outputs=[results_out, reward_plot, win_plot, graph_table_out],
    )

if __name__ == "__main__":
    demo.launch()
