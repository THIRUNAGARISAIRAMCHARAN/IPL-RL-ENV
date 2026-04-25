from huggingface_hub import HfApi
import os

api = HfApi()
token = os.getenv("HF_TOKEN")
repo_id = "thirunagarisairamcharan/IPL-RL-ENV"

# We will try both lowercase and uppercase username if one fails
files_to_upload = ["app.py", "README.md", "requirements.txt"]

# Also include the scripts folder
scripts = ["scripts/hf_train.py", "scripts/generate_reward_curve.py", "scripts/push_to_hub.py", "scripts/get_proof.py"]

all_files = files_to_upload + scripts

print(f"Starting sync into {repo_id}...")
for f in all_files:
    if os.path.exists(f):
        print(f"Uploading {f}...")
        try:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="space",
                token=token
            )
        except Exception as e:
            print(f"Failed to upload {f} to {repo_id}: {e}")
            # Try uppercase fallback if it was a 404/401 that might be username case related
            repo_id_alt = "THIRUNAGARISAIRAMCHARAN/IPL-RL-ENV"
            print(f"Retrying with {repo_id_alt}...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=repo_id_alt,
                repo_type="space",
                token=token
            )
            # Update repo_id for subsequent files if successful
            repo_id = repo_id_alt

print("Sync complete.")
