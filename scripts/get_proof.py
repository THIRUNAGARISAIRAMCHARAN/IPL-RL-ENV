import sys
import os

# Add root to path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training.reward_logger import RewardLogger

def main():
    print("--- LEARNING PROOF DATA ---")
    try:
        logger = RewardLogger()
        proof = logger.get_learning_proof()
        
        print("\nCopy the following into BLOG.md or PROJECT_REPORT.md:\n")
        print("```json")
        print(proof)
        print("```")
        
    except Exception as e:
        print(f"Error generating proof: {e}")
        print("Make sure training/logs/rewards.csv exists.")

if __name__ == "__main__":
    main()
