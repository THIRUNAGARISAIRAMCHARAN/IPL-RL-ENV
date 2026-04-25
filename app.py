import gradio as gr
import json
import os
import pandas as pd
import threading
from env.ipl_env import IPLAuctionEnv
from agents.base_agent import BaseIPLAgent
from training.train import run_training

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
training_status = {"ongoing": False, "msg": "Ready to learn."}

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
    squads_df = pd.DataFrame(squad_list, columns=["Team", "Player", "Role", "Price"]).sort_values("Team")

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
    top_4 = standings_df["Team"].tolist()[:4]
    qualification_text += f"The top 4 teams qualified: **{', '.join(top_4)}**\n\n"
    
    bracket_text = "### 🏆 Playoff Brackets\n"
    if bracket:
        bracket_text += f"- **Qualifier 1**: {bracket['Q1']['winner']} defeated {bracket['Q1']['loser']}\n"
        bracket_text += f"- **Eliminator**: {bracket['Eliminator']['winner']} defeated {bracket['Eliminator']['loser']}\n"
        bracket_text += f"- **Qualifier 2**: {bracket['Q2']['winner']} defeated {bracket['Q2']['loser']}\n"
        bracket_text += f"- **Grand Final**: {bracket['Final']['winner']} defeated {bracket['Final']['loser']}\n"
    
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
        "\n".join(log[-30:]) if log else "No lot-close events captured.", 
        squads_df,
        standings_df,
        matches_df,
        qualification_text,
        bracket_text,
        champ_text,
        transfer_text
    )

def start_training_ui(episodes):
    if training_status["ongoing"]:
        return "Training is already in progress in the background."
    
    def worker():
        training_status["ongoing"] = True
        training_status["msg"] = f"Learning... (Target: {episodes} episodes)"
        try:
            run_training(episodes=int(episodes))
            training_status["msg"] = f"Success! Completed {episodes} episodes. Refresh logs to see results."
        except Exception as e:
            training_status["msg"] = f"Early stop or Error: {str(e)}"
        finally:
            training_status["ongoing"] = False

    thread = threading.Thread(target=worker)
    thread.start()
    return "Training started! You can check the 'Training Metrics' tab in a few minutes to see progress. The simulation will continue to work while training runs."

def get_training_status():
    return training_status["msg"]

def load_results():
    try:
        if not os.path.exists("training/logs/reward_curve.json"):
            return "No training data yet. Pull from Hub or start training."
        with open("training/logs/reward_curve.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        teams = data.get("teams", {})
        lines = []
        for team, metrics in teams.items():
            rewards = metrics.get("rewards", [])
            if rewards:
                window = rewards[-10:] if len(rewards) >= 10 else rewards
                avg = sum(window) / max(1, len(window))
                lines.append(f"{team}: Avg Reward (last 10 eps) = {avg:.1f}")
        return "\n".join(lines) or "No training data yet. Pull from Hub or start training."
    except Exception:
        return "No training data yet. Pull from Hub or start training."


with gr.Blocks(title="IPL RL Auction Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPL Multi-Agent RL Auction Environment")
    gr.Markdown("Teaching 8 AI agents to draft, manage, and optimize championship-winning squads across 3 phases.")
    
    with gr.Tab("Phase 1: Auction"):
        with gr.Row():
            with gr.Column(scale=1):
                run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
                auction_log = gr.Textbox(label="Live Auction Bidding (Last 30 Events)", lines=10)
            with gr.Column(scale=2):
                squads_out = gr.Dataframe(label="Final Rosters (All Players)")
    
    with gr.Tab("Phase 2: Season results"):
        gr.Markdown("### Simulate 56 matches, playoffs, and crown a champion")
        with gr.Row():
            with gr.Column():
                standings_out = gr.Dataframe(label="Final League Table (14 matches each)")
            with gr.Column():
                matches_out = gr.Dataframe(label="Full Season Match List (56 Matches)")
        
        with gr.Row():
            with gr.Column():
                qual_out = gr.Markdown("### Playoff Qualification")
            with gr.Column():
                bracket_out = gr.Markdown("### Playoff Road to Final")
        
        champ_out = gr.Markdown("## FINAL WINNER: No simulation run yet.")
        
    with gr.Tab("Phase 3: Transfer Window"):
        transfer_out = gr.Markdown("No transfer activity logged yet.")

    with gr.Tab("AI Learning Center"):
        gr.Markdown("### Train the Agents (No Terminal Required)")
        with gr.Row():
            eps_input = gr.Slider(minimum=10, maximum=500, value=200, step=10, label="Episodes to Train")
            train_btn = gr.Button("🚀 Start AI Learning Session", variant="secondary")
        
        train_status_out = gr.Textbox(label="Current Status", value="Ready.")
        gr.Markdown("> **Note**: On Hugging Face, training 200 episodes takes ~20 minutes. You can refresh the 'Training Metrics' tab periodically to see rewards increasing.")
        
        train_btn.click(fn=start_training_ui, inputs=eps_input, outputs=train_status_out)
        gr.Button("Check Progress").click(fn=get_training_status, outputs=train_status_out)

    with gr.Tab("Training Metrics"):
        gr.Markdown("### Reward Progression Across Episodes")
        results_out = gr.Textbox(label="Last 10 Episode Averages", lines=12)
        gr.Button("Refresh Logs").click(fn=load_results, outputs=results_out)
        
    with gr.Tab("About"):
        gr.Markdown("""
        ### Project Context
        This environment tests **Long-Horizon Planning** and **Multi-Agent Interaction**. 
        
        - **Phase 1: Auction** — Agents bid for 200 players with hidden stats and noisy budget tracking.
        - **Phase 2: Season** — Simulate 56 matches, playoffs, and crown a champion.
        - **Phase 3: Transfer** — Agents attempt mid-season recovery through strategic trades.
        
        Built using the **OpenEnv** framework for high-fidelity RL orchestration.
        """)

    run_btn.click(
        fn=run_demo_auction, 
        outputs=[auction_log, squads_out, standings_out, matches_out, qual_out, bracket_out, champ_out, transfer_out]
    )

if __name__ == "__main__":
    demo.launch()
