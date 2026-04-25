import gradio as gr
import json
import os
import pandas as pd
from env.ipl_env import IPLAuctionEnv
from agents.base_agent import BaseIPLAgent

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

    # Squads
    squads = last_info.get("final_squads", env.team_squads)
    squad_text = ""
    for tid, squad in squads.items():
        names = ", ".join([p["name"] for p in squad[:5]])
        squad_text += f"**{str(tid)}**: {names}...\n"

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
    
    champion = results.get("champion", "N/A")
    champ_text = f"## TOURNAMENT CHAMPION: {champion}"

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
        squad_text,
        standings_df,
        champ_text,
        transfer_text
    )


def load_results():
    try:
        if not os.path.exists("training/logs/reward_curve.json"):
            return "No training data yet. Run train.py first."
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
        return "\n".join(lines) or "No training data yet. Run train.py first."
    except Exception:
        return "No training data yet. Run train.py first."


with gr.Blocks(title="IPL RL Auction Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IPL Multi-Agent RL Auction Environment")
    gr.Markdown("Teaching 8 AI agents to draft, manage, and optimize championship-winning squads across 3 phases.")
    
    with gr.Tab("Phase 1: Auction"):
        with gr.Row():
            with gr.Column(scale=2):
                run_btn = gr.Button("Run Full Simulation Cycle", variant="primary")
                auction_log = gr.Textbox(label="Live Auction Bidding (Last 30 Events)", lines=15)
            with gr.Column(scale=1):
                squads_out = gr.Markdown(label="Final Squad Fragments")
    
    with gr.Tab("Phase 2: Season results"):
        champ_out = gr.Markdown("### No simulation run yet.")
        standings_out = gr.Dataframe(label="Final League Table (14 matches each)")
        
    with gr.Tab("Phase 3: Transfer Window"):
        transfer_out = gr.Markdown("No transfer activity logged yet.")

    with gr.Tab("Training Metrics"):
        gr.Markdown("### Reward Progression Across Episodes")
        results_out = gr.Textbox(label="Last 10 Episode Averages", lines=12)
        gr.Button("Refresh Logs").click(fn=load_results, outputs=results_out)
        
    with gr.Tab("About"):
        gr.Markdown("""
        ### Project Context
        This environment tests **Long-Horizon Planning** and **Multi-Agent Interaction**. 
        
        - **Phase 1: Auction** — Agents bid for 200 players with hidden stats and noisy budget tracking.
        - **Phase 2: Season** — Roster quality is tested across a 56-match simulated league.
        - **Phase 3: Transfer** — Agents attempt mid-season recovery through strategic trades.
        
        Built using the **OpenEnv** framework for high-fidelity RL orchestration.
        """)

    run_btn.click(
        fn=run_demo_auction, 
        outputs=[auction_log, squads_out, standings_out, champ_out, transfer_out]
    )

if __name__ == "__main__":
    demo.launch()
