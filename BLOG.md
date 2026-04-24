# IPL RL: Teaching 8 AI Agents to Build Championship Teams

## The Problem
IPL auctions are deceptively hard. Every franchise starts with Rs.90 Crore, but that number disappears quickly when marquee players trigger bidding wars. The catch is that auction decisions are made under uncertainty: agents can see visible stats, but true player form and injury risk are hidden variables. They also do not get perfect market information, because opponent budgets are observed with noise. In practice, this means each bid is both a valuation decision and a game-theory decision.

The challenge becomes even harder because auction mistakes compound over time. Overpaying for two stars can leave a squad unbalanced, and that weakness only becomes obvious after a full 14-match league phase. Most naive policies panic-bid, over-index on brand names, and run out of flexibility before the season starts.

## The 3-Phase Environment
**Phase 1: Auction.** Eight agents (MI, CSK, RCB, KKR, DC, RR, PBKS, SRH) bid in a partially observable market. Opponent budgets are noisy estimates (+-20%), player true form is hidden, and teams follow one of four personalities: aggressive, conservative, balanced, or role_filler.

**Phase 2: Season.** A full round-robin (56 matches) converts auction quality into outcomes. The environment simulates form variation, injury effects, squad balance, and matchup randomness. Teams that spend emotionally in auction often carry structural weaknesses and can finish in the bottom two despite signing stars.

**Phase 3: Transfer.** A mid-season correction window allows up to two trades per team. This tests whether agents can recover from poor initial allocation and patch role gaps instead of doubling down on sunk-cost behavior.

## 14 Reward Signals
Learning is shaped by dense, interpretable rewards: `value_pick`, `synergy`, `late_bonus`, `panic_penalty`, `block_reward`, `waste_penalty`, `balance_bonus`, plus season win rewards, upset bonus, playoff/top-table bonuses, champion bonus, and transfer-quality signals. Every component is logged per team per episode, enabling forensic analysis instead of black-box scoring.

## What Agents Learned
Before training, agents routinely panic-bid on marquee players, overpaid above intrinsic value, and entered the season with weak role coverage.

After tracked runs, current logged insights from `emergent_insights.json` are:
- `reward_improvement_pct`: 0.0
- `win_rate_improvement_pct`: 0.0
- `budget_efficiency_improvement_pct`: 0.0

These values indicate baseline/noisy initial runs and provide a clean checkpoint before longer training. As episodes accumulate, this section will be replaced with behavior-shift narratives from emergent insights.

## Results
Reward +0.0%  |  Win Rate +0.0%  |  Budget Efficiency +0.0%

[EMBED reward_curve.png HERE after onsite training]

## Themes
#1 Multi-Agent  |  #2 Long-Horizon  |  #4 Self-Improvement

---

## 2-MINUTE VIDEO SCRIPT
[0:00-0:12] HOOK: "Most AI is told what to do. We built 8 agents that learn IPL auction strategy from scratch."

[0:12-0:35] PROBLEM: "Winning IPL is budget management + opponent modeling + living with auction mistakes for 14 matches."

[0:35-1:05] DEMO: Show HF Spaces live auction. Point to bluff detection.

[1:05-1:35] RESULTS: Before/After panel - panic rate, win rate, efficiency.

[1:35-2:00] CTA: "Try the demo - link in description."
