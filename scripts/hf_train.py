import os
import sys
from huggingface_hub import login
from training.train import run_training

def main():
    # 1. Login using HF_TOKEN
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)
    
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    # 2. Check for Weights & Biases
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            print("WandB login successful.")
        except ImportError:
            print("WandB not installed, skipping.")
    else:
        print("WANDB_API_KEY not found, skipping WandB logging.")

    # 3. Create required directories
    os.makedirs("training/logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 4. Resolve Episodes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    args = parser.parse_args()

    print(f"Starting Training (Target {args.episodes} episodes)...")
    try:
        run_training(episodes=args.episodes)
    except Exception as e:
        print(f"Training interrupted: {e}")
        if args.episodes > 200:
            print("Attempting to run with fallback settings (200 episodes)...")
            run_training(episodes=200)

if __name__ == "__main__":
    main()
