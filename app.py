import gradio as gr
import json
import os
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

    squads = last_info.get("final_squads", env.team_squads)
    squad_text = ""
    for tid, squad in squads.items():
        names = ", ".join([p["name"] for p in squad[:5]])
        squad_text += f"{str(tid)}: {names}...\n"
    return "\n".join(log[-30:]) if log else "No lot-close events captured.", squad_text


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


with gr.Blocks(title="IPL RL Auction") as demo:
    gr.Markdown("# 🏏 IPL Multi-Agent RL Auction Environment")
    with gr.Tab("Try It"):
        run_btn = gr.Button("Run Auction Episode", variant="primary")
        auction_log = gr.Textbox(label="Auction Log (last 30)", lines=15)
        squads_out = gr.Textbox(label="Final Squads", lines=10)
        run_btn.click(fn=run_demo_auction, outputs=[auction_log, squads_out])
    with gr.Tab("Training Results"):
        results_out = gr.Textbox(label="Training Results", lines=12)
        gr.Button("Load Results").click(fn=load_results, outputs=results_out)
    with gr.Tab("About"):
        gr.Markdown("Multi-agent RL for IPL team building. Theme #1 #2 #4.")

if __name__ == "__main__":
    demo.launch()
