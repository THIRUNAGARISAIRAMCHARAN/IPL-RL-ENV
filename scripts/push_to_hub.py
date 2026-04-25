from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def _latest_checkpoint_dir(root: Path) -> Path | None:
    checkpoints_dir = root / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = [p for p in checkpoints_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Push IPL RL artifacts to HuggingFace Hub.")
    parser.add_argument("--username", required=True, help="Your HuggingFace username.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "training" / "logs"
    reward_curve_png = root / "reward_curve.png"

    model_repo = f"{args.username}/ipl-rl-agent"
    space_repo = f"{args.username}/ipl-rl-env"

    api = HfApi()

    print(f"[1/5] Creating/updating model repo: {model_repo}")
    api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)

    latest_ckpt = _latest_checkpoint_dir(root)
    if latest_ckpt is not None:
        print(f"[2/5] Uploading model checkpoint folder: {latest_ckpt}")
        api.upload_folder(
            repo_id=model_repo,
            repo_type="model",
            folder_path=str(latest_ckpt),
            path_in_repo=".",
        )
    else:
        print("[2/5] No checkpoint folder found. Uploading placeholder model card.")
        tmp_model_card = root / "MODEL_CARD.md"
        tmp_model_card.write_text("# IPL RL Agent\n\nUpload a trained checkpoint to this repo.\n", encoding="utf-8")
        api.upload_file(
            repo_id=model_repo,
            repo_type="model",
            path_or_fileobj=str(tmp_model_card),
            path_in_repo="README.md",
        )

    print(f"[3/5] Creating/updating Space repo: {space_repo}")
    api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)

    if logs_dir.exists():
        print(f"[4/5] Uploading logs folder: {logs_dir}")
        api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=str(logs_dir),
            path_in_repo="training/logs",
        )
    else:
        print("[4/5] Logs folder not found, skipping folder upload.")

    if reward_curve_png.exists():
        print(f"[5/5] Uploading reward curve image: {reward_curve_png}")
        api.upload_file(
            repo_id=space_repo,
            repo_type="space",
            path_or_fileobj=str(reward_curve_png),
            path_in_repo="reward_curve.png",
        )
    else:
        print("[5/5] reward_curve.png not found, skipping image upload.")

    print("Hub push complete.")


if __name__ == "__main__":
    main()
