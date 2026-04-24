import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="IPL RL Dashboard", layout="wide", page_icon="🏏")

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


# Safe loader — always returns default, never crashes
def load_json(fname, default):
    try:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _team_label_from_winner(raw):
    if isinstance(raw, int) and 0 <= raw < len(TEAM_NAMES):
        return TEAM_NAMES[raw]
    raw_str = str(raw)
    if raw_str.isdigit():
        idx = int(raw_str)
        if 0 <= idx < len(TEAM_NAMES):
            return TEAM_NAMES[idx]
    if raw_str in TEAM_NAMES:
        return raw_str
    return "--"


panel = st.sidebar.radio(
    "Panel",
    [
        "Live Auction",
        "Team Panels",
        "Learning Graphs",
        "Season Results",
        "Before vs After",
        "Strategy Insights",
    ],
)

auto_refresh = st.sidebar.checkbox("Auto-refresh (2s)", value=False)
if auto_refresh:
    import time

    time.sleep(2)
    st.rerun()

# PANEL 1: LIVE AUCTION
if panel == "Live Auction":
    st.title("Live Auction Feed")
    auction_log = load_json("training/logs/auction_log.json", [])
    if not auction_log:
        st.warning("No auction data yet. Run: python training/train.py")
    else:
        last = auction_log[-1]
        col1, col2 = st.columns([4, 6])
        with col1:
            st.metric("Player", last.get("player_name", "--"))
            st.metric("Current Bid", f"Rs.{last.get('price', 0):.1f} Cr")
            st.metric("Leader", _team_label_from_winner(last.get("winner", 0)))
            if last.get("bluff"):
                st.warning("Bluff detected!")
        with col2:
            squads = load_json("training/logs/squads.json", {})
            budgets = {}
            for i, t in enumerate(TEAM_NAMES):
                team_data = squads.get(str(i), {})
                if isinstance(team_data, dict):
                    budgets[t] = float(team_data.get("budget_remaining", 90))
                else:
                    budgets[t] = 90.0
            fig = px.bar(
                x=list(budgets.keys()),
                y=list(budgets.values()),
                color=list(budgets.values()),
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Last 20 Lots")
        for lot in reversed(auction_log[-20:]):
            w = _team_label_from_winner(lot.get("winner", 0))
            st.write(f"{lot.get('player_name', '?')} -> {w} @ Rs.{lot.get('price', 0):.1f}Cr")

# PANEL 2: TEAM PANELS
elif panel == "Team Panels":
    st.title("IPL Team Panels")
    squads = load_json("training/logs/squads.json", {})
    cols = st.columns(4)
    for i, team in enumerate(TEAM_NAMES):
        with cols[i % 4]:
            data = squads.get(str(i), {})
            players = data.get("players", []) if isinstance(data, dict) else []
            budget = float(data.get("budget_remaining", 90)) if isinstance(data, dict) else 90.0
            st.markdown(f"**{team}**")
            st.caption(f"Budget: Rs.{budget:.1f}Cr  | Squad: {len(players)}")
            if players:
                df = pd.DataFrame(
                    [{"Name": p.get("name", "?"), "Role": p.get("role", "?"), "Paid": p.get("price_paid", 0)} for p in players[:8]]
                )
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.info("No squad data yet.")

# PANEL 3: LEARNING GRAPHS
elif panel == "Learning Graphs":
    st.title("Learning Graphs")
    curve = load_json("training/logs/reward_curve.json", {})
    if not curve.get("episodes"):
        st.info("No training data yet. Run training/train.py to populate.")
    else:
        episodes = curve["episodes"]
        teams_data = curve.get("teams", {})
        fig1 = go.Figure()
        for team, metrics in teams_data.items():
            fig1.add_trace(
                go.Scatter(
                    x=episodes,
                    y=metrics.get("rewards", []),
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )
        fig1.update_layout(title="Reward per Episode", xaxis_title="Episode")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        for team, metrics in teams_data.items():
            fig2.add_trace(
                go.Scatter(
                    x=episodes,
                    y=metrics.get("win_rate", []),
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )
        fig2.update_layout(title="Win Rate vs Episodes", xaxis_title="Episode")
        st.plotly_chart(fig2, use_container_width=True)

        from training.reward_logger import RewardLogger

        proof = RewardLogger().get_learning_proof()
        st.success(
            f"Reward +{proof.get('reward_improvement_pct', 0):.1f}%  |  "
            f"Win Rate +{proof.get('win_rate_improvement_pct', 0):.1f}%"
        )

elif panel == "Season Results":
    st.title("Season Standings")
    season = load_json("training/logs/season_results.json", {})
    if not season.get("standings"):
        st.info("Season not yet simulated. Run at least 1 training episode.")
    else:
        standings = season["standings"]
        rows = []
        for team_id, stats in standings.items():
            tid = int(team_id) if str(team_id).isdigit() else TEAM_NAMES.index(team_id) if team_id in TEAM_NAMES else 0
            rows.append(
                {
                    "Team": TEAM_NAMES[tid],
                    "Wins": stats.get("wins", 0),
                    "Losses": stats.get("losses", 0),
                    "NRR": round(stats.get("nrr", 0), 3),
                }
            )
        df = pd.DataFrame(rows).sort_values("Wins", ascending=False)
        champion = season.get("champion")
        if champion is not None:
            champ_name = TEAM_NAMES[int(champion)] if str(champion).isdigit() else str(champion)
            st.success(f"Champion: {champ_name}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        bracket = season.get("bracket", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Q1**")
            st.write(bracket.get("q1", bracket.get("Q1", "TBD")))
        with c2:
            st.write("**Eliminator + Q2**")
            elim = bracket.get("eliminator", bracket.get("Eliminator", "TBD"))
            q2 = bracket.get("q2", bracket.get("Q2", "TBD"))
            st.write(f"{elim}\n\n{q2}")
        with c3:
            st.write("**FINAL**")
            st.write(bracket.get("final", bracket.get("Final", "TBD")))

# PANEL 5: BEFORE vs AFTER
elif panel == "Before vs After":
    st.title("Before vs After Training")
    behaviors = load_json("training/logs/behavior_summaries.json", [])
    if isinstance(behaviors, dict):
        # Support current logger format where latest summary may be a dict.
        behaviors = [behaviors]
    if len(behaviors) < 20:
        st.info("Need at least 20 training episodes.")
    else:
        n = len(behaviors)
        slider = st.slider("Compare episodes:", 10, n, (10, n), step=10)
        early = behaviors[: slider[0]]
        late = behaviors[slider[1] - 10 : slider[1]]

        def avg_metric(eps, metric):
            vals = [ep.get(str(i), {}).get(metric, 0) for ep in eps for i in range(8)]
            return sum(vals) / len(vals) if vals else 0

        metrics = ["overbid_rate", "block_rate", "patience_score", "bluff_success_rate"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"EARLY (ep 1-{slider[0]})")
            for m in metrics:
                st.metric(m.replace("_", " ").title(), f"{avg_metric(early, m):.2%}")
        with c2:
            st.subheader(f"LATE (ep {slider[1]-10}-{slider[1]})")
            for m in metrics:
                ev, lv = avg_metric(early, m), avg_metric(late, m)
                st.metric(m.replace("_", " ").title(), f"{lv:.2%}", delta=f"{lv-ev:+.2%}")

# PANEL 6: STRATEGY INSIGHTS
elif panel == "Strategy Insights":
    st.title("Strategy Insights")
    insights = load_json("training/logs/emergent_insights.json", [])
    behaviors = load_json("training/logs/behavior_summaries.json", [])
    if isinstance(behaviors, dict):
        behaviors = [behaviors]

    if isinstance(insights, dict):
        insights_list = [f"{k}: {v}" for k, v in insights.items()]
    else:
        insights_list = insights

    if not insights_list:
        st.info("No insights yet. Need 25+ training episodes.")
    else:
        st.subheader("What the Agents Learned")
        for insight in insights_list:
            st.write(f"• {insight}")
    if behaviors:
        st.subheader("Agent Strategy Evolution")
        for tid, team in enumerate(TEAM_NAMES):
            labels = [ep.get(str(tid), {}).get("label", "?") for ep in behaviors[::10]]
            st.write(f"**{team}**: " + " → ".join(labels[-8:]))
    with st.sidebar:
        st.markdown("---")
        st.subheader("Recent Lots")
        auction_log = load_json("training/logs/auction_log.json", [])
        for lot in auction_log[-10:]:
            st.write(f"{lot.get('player_name', '?')} @ Rs.{lot.get('price', 0):.1f}Cr")
