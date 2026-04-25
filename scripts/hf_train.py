import os
import sys
from huggingface_hub import login

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    os.chdir(ROOT_DIR)

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

    episodes = max(200, int(args.episodes))
    print(f"Starting Training (Target {episodes} episodes)...")
    run_training = None
    try:
        from training.train import run_training

        run_training(episodes=episodes)
    except Exception as e:
        print(f"Training interrupted: {e}")
        if episodes > 200 and run_training is not None:
            print("Attempting to run with fallback settings (200 episodes)...")
            run_training(episodes=200)

if __name__ == "__main__":
    main()
