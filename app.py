import gradio as gr
import json
import os
import pandas as pd
import threading
import sys
from env.ipl_env import IPLAuctionEnv
from agents.base_agent import BaseIPLAgent
from training.train import run_training

def _start_training():
    try:
        run_training(episodes=50)
    except Exception as e:
        print(f"Training error: {e}", flush=True)

# Only start once
try:
    import streamlit as st
    _training_state = st.session_state
except Exception:
    _training_state = {}

if "training_started" not in _training_state:
    _training_state["training_started"] = True
    t = threading.Thread(target=_start_training, daemon=True)
    t.start()

TEAM_NAMES = ["MI", "CSK", "RCB", "KKR", "DC", "RR", "PBKS", "SRH"]
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

# Global status for training
training_status = {"ongoing": False, "msg": "Ready."}

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
    squad_list = []
    for tid, squad in squads.items():
        for p in squad:
            squad_list.append([
                tid,
                p.get("name", "Unknown"),
                p.get("role", "N/A"),
                f"Rs.{p.get('price', 0):.1f}Cr"
            ])
    if squad_list:
        squads_df = pd.DataFrame(squad_list, columns=["Team", "Player", "Role", "Price"]).sort_values("Team")
    else:
        squads_df = pd.DataFrame(columns=["Team", "Player", "Role", "Price"])

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
    standings_df = pd.DataFrame(standings_data, columns=["Rank", "Team", "Wins", "Losses", "NRR"]).sort_values("Rank")
    
    # All 56 Matches
    match_list = []
    for i, m in enumerate(results.get("results", []), 1):
        match_list.append([
            f"M{i}",
            f"{m['team_a']} vs {m['team_b']}",
            m['winner'],
            "Yes" if m.get("upset") else "No"
        ])
    matches_df = pd.DataFrame(match_list, columns=["Match ID", "Fixture", "Winner", "Upset"])

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
        squads_df,
        standings_df,
        matches_df,
        qualification_text,
        bracket_text,
        champ_text,
        transfer_text
    )

def start_training_ui():
    episodes = 200
    if training_status["ongoing"]:
        return "Training is already running."
    
    def worker():
        training_status["ongoing"] = True
        training_status["msg"] = "Learning (200 episodes)..."
        try:
            run_training(episodes=200)
            training_status["msg"] = "Done! 200 episodes completed."
        except Exception as e:
            training_status["msg"] = f"Stopped: {str(e)}"
        finally:
            training_status["ongoing"] = False

    threading.Thread(target=worker).start()
    return "Started 200-episode training in the background."

def get_training_status():
    return training_status["msg"]

def load_results():
    try:
        path = "training/logs/reward_curve.json"
        if not os.path.exists(path):
            return "No training logs found."
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = data.get("teams", {})
        lines = []
        for team, metrics in teams.items():
            rewards = metrics.get("rewards", [])
            if rewards:
                avg = sum(rewards[-10:]) / max(1, len(rewards[-10:]))
                lines.append(f"{team}: Avg Reward = {avg:.1f}")
        return "\n".join(lines) or "No data."
    except:
        return "No training data available."


with gr.Blocks(title="IPL RL Auction Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPL Multi-Agent RL Auction Environment")
    
    with gr.Tab("Phase 1: Auction"):
        with gr.Row():
            with gr.Column(scale=1):
                run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
                auction_log = gr.Textbox(label="Live Log", lines=10)
            with gr.Column(scale=2):
                squads_out = gr.Dataframe(label="Final Rosters")
    
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

    with gr.Tab("AI Learning Center"):
        gr.Markdown("### Train the Agents")
        train_btn = gr.Button("🚀 Start 200 Episode Training Run", variant="secondary")
        train_status_out = gr.Textbox(label="Status", value="Ready.")
        gr.Button("Refresh Status").click(fn=get_training_status, outputs=train_status_out)
        train_btn.click(fn=start_training_ui, outputs=train_status_out)

    with gr.Tab("Training Metrics"):
        results_out = gr.Textbox(label="Reward Stats", lines=10)
        gr.Button("Refresh Logs").click(fn=load_results, outputs=results_out)
        
    with gr.Tab("About"):
        gr.Markdown("Environment for Multi-Agent IPL Auction RL.")

    run_btn.click(
        fn=run_demo_auction, 
        outputs=[auction_log, squads_out, standings_out, matches_out, qual_out, bracket_out, champ_out, transfer_out]
    )

if __name__ == "__main__":
    demo.launch()
