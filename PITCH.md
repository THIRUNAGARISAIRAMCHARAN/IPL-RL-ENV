# 30-SECOND PITCH (memorise word for word)
We built a multi-agent RL environment where 8 AI agents —
representing MI, CSK, RCB, KKR, DC, RR, PBKS, and SRH —
compete to build championship-winning teams.
Agents model opponents, detect bluffs, manage 90-crore budgets,
and live with their auction mistakes across a full IPL season.
After training, agents evolved from reckless bidders to strategic
planners — proven by reward curves, win rates, and budget efficiency.
Built on OpenEnv. Trained with HuggingFace TRL.

## JUDGE Q&A — PREPARE ALL 6 ANSWERS

**Q1: Why IPL?**  
Budget constraints + partial observability + delayed rewards + 8 agents.
Instantly relatable. Data is real.

**Q2: How is this different from a basic auction sim?**  
3 phases not 1. Hidden info. 4 personality types. Bluffing.
Season outcomes DEPEND on auction quality — true long-horizon causal RL.

**Q3: Is the LLM reasoning or pattern matching?**  
Emergent behaviors — blocking, bluff detection, patience — are NOT
hard-coded. They emerge from reward signals. Our behavior_summary logs prove it.

**Q4: What was hardest to build?**  
Getting information hiding right. +-20% budget noise was the calibration.
Too much signal = trivial. Too little = unlearnable.

**Q5: What would you add with more time?**  
Mid-season player injuries, fan sentiment budgets, historical data pre-training.

**Q6: Why OpenEnv?**  
Standardized interface. Shareable, reproducible, comparable.
`validate()` ensures correctness. Other researchers can swap in agents.

## DEMO ORDER (practice this sequence)
1. Open dashboard Panel 1 (Live Auction) — show real IPL names
2. Switch to Panel 3 (Learning Graphs) — point to upward reward curve
3. Switch to Panel 5 (Before vs After) — show panic rate dropped
4. Switch to Panel 6 (Strategy Insights) — read one emergent behavior aloud
5. Open HuggingFace Space — click Run Auction Episode

## DO NOT SAY vs SAY
**DO NOT SAY:** "We built an IPL auction simulation"  
**SAY:** "A multi-agent RL environment where LLM agents learn strategic
bidding, opponent modeling, and long-term team optimization under
uncertainty — spanning auction, season, and transfer phases."
