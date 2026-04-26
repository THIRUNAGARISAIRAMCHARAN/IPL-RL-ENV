---
title: IPL Multi-Agent RL Auction Environment
emoji: 🏏
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.10.0
app_file: app.py
pinned: true
tags:
  - reinforcement-learning
  - multi-agent
  - ipl
  - grpo
  - openenv
  - long-horizon-planning
---

# 🏏 IPL Multi-Agent RL Auction Environment

[![HuggingFace Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/thirunagarisairamcharan/meta-ipl-hackathon)

## Problem

The IPL auction is one of cricket's most complex strategic events. 8 franchises compete to build the best squad within a fixed 90 Crore budget — with hidden opponent budgets, uncertain player values, and delayed rewards that only materialize after 56 season matches.

This makes it a perfect environment for training LLMs on:
- **Theme 1: Multi-Agent Interactions** — 8 agents competing simultaneously with hidden information
- **Theme 2: Long-Horizon Planning** — Auction decisions in Phase 1 only reward in Phase 2

## Environment

The agent sees:
- Current player (role, tier, base price)
- Own budget remaining
- Own squad composition
- Players remaining in auction

The agent can:
- `bid(amount)` — bid on the current player
- `pass_player()` — skip and move to next player

Rewards are shaped by:
- Value picks (buying below true value)
- Squad balance (correct role distribution)
- Budget efficiency
- Season performance

## Training

Trained using **HuggingFace TRL GRPOTrainer** with `Qwen/Qwen2.5-0.5B-Instruct` for 50 episodes.

### Reward Curve
![Reward Curve](training/plots/reward_curve.png)

### Win Rate
![Win Rate](training/plots/win_rate.png)

### Before vs After
![Before vs After](training/plots/before_after.png)

## Results

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Episodes | 50 |
| Final Reward | 1.0 |
| Train Loss | 0.00689 |
| Framework | HuggingFace TRL (GRPO) |

## Links

- 🚀 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/thirunagarisairamcharan/meta-ipl-hackathon)
- 📝 **Blog Post:** [HuggingFace Blog](https://huggingface.co/blog/thirunagarisairamcharan/ipl-rl-environment)
- 📓 **Colab Notebook:** [Run Training](https://colab.research.google.com/drive/your-notebook-link)

## How to Run

```bash
pip install -r requirements.txt
python training/train.py --episodes 50
```

## OpenEnv Compliance

This environment follows OpenEnv standards:
- ✅ Extends `Environment` base class
- ✅ Implements `reset()`, `bid()`, `pass_player()` tools
- ✅ Valid `openenv.yaml` manifest
- ✅ Hosted on HuggingFace Spaces
