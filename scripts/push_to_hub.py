import argparse
import os
from huggingface_hub import HfApi, login

def push_results(username):
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found. Use 'huggingface-cli login' or set HF_TOKEN env var.")
        return

    api = HfApi()
    
    # 1. Push Model
    model_repo = f"{username}/ipl-rl-agent"
    print(f"Pushing model to {model_repo}...")
    try:
        # Pushing the best checkpoint or the main checkpoints folder
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            latest_cp = sorted(os.listdir(checkpoint_dir))[-1]
            api.upload_folder(
                folder_path=os.path.join(checkpoint_dir, latest_cp),
                repo_id=model_repo,
                repo_type="model"
            )
            print("Model pushed successfully.")
        else:
            print("No checkpoints found to push.")
    except Exception as e:
        print(f"Model push failed: {e}")

    # 2. Push Logs to Space
    space_repo = f"{username}/IPL-RL-ENV"
    print(f"Uploading logs to Space {space_repo}...")
    try:
        # Upload the logs folder
        api.upload_folder(
            folder_path="training/logs",
            path_in_repo="training/logs",
            repo_id=space_repo,
            repo_type="space"
        )
        # Upload the reward curve image
        if os.path.exists("reward_curve.png"):
            api.upload_file(
                path_or_fileobj="reward_curve.png",
                path_in_repo="reward_curve.png",
                repo_id=space_repo,
                repo_type="space"
            )
        print("Logs and plots pushed to Space successfully.")
    except Exception as e:
        print(f"Space upload failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    args = parser.parse_args()
    push_results(args.username)
