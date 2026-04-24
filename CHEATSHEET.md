# IPL RL Hackathon Stage Cheat Sheet

## 30-Second Pitch (Memory Bullets)
- 8 AI agents (MI, CSK, RCB, KKR, DC, RR, PBKS, SRH) compete in one RL environment.
- They bid under uncertainty: hidden player form, noisy opponent budgets, bluff signals.
- Decisions carry across auction -> season -> transfer (true long-horizon learning).
- After training: more strategic bidding, better win rate, stronger budget efficiency.

## Demo Order (Follow Exactly)
1. Open Dashboard -> **Panel 1: Live Auction** (show real IPL names and bids).
2. Go to **Panel 3: Learning Graphs** (show reward curve trend).
3. Go to **Panel 5: Before vs After** (show panic/overbid drop).
4. Go to **Panel 6: Strategy Insights** (read one emergent behavior line).
5. Open HuggingFace Space -> click **Run Auction Episode**.
6. Close with outcomes: reward improvement, win-rate improvement, budget efficiency.

## Judge Q&A (Short Answers)
- **Q:** Why IPL?  
  **A:** Real, relatable, and complex: budget limits + hidden info + delayed rewards + 8-agent competition.

- **Q:** How is this different from a basic auction sim?  
  **A:** It is 3 phases, not 1. Season outcomes depend on auction quality, so strategy is long-horizon.

- **Q:** Is the LLM reasoning or pattern matching?  
  **A:** Behaviors like blocking, patience, and bluff response emerge from rewards, not hard-coded rules.

- **Q:** What was hardest to build?  
  **A:** Information hiding calibration. +-20% budget noise had to be informative but not game-breaking.

- **Q:** What would you add with more time?  
  **A:** Injury shocks, fan-sentiment budget effects, and historical-data pretraining.

- **Q:** Why OpenEnv?  
  **A:** Standard interface, reproducible experiments, and `validate()` checks for environment correctness.

## DO Say / DON'T Say
| DO say | DON'T say |
|---|---|
| Multi-agent RL environment with long-horizon decision-making | Just an IPL auction simulator |
| Agents learn bidding, opponent modeling, and recovery via transfers | We only optimize one auction round |
| Emergent behavior is measured with logged reward/strategy signals | The model just guesses randomly |

## If It Crashes (Emergency Fallback)
- "Even if live UI fails, we have logged outputs: reward curves, behavior summaries, and season results proving learning over episodes."
