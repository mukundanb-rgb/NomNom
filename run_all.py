"""One command to rebuild everything from scratch, in order.

    python run_all.py          # full rebuild
    python run_all.py --quick  # smaller dataset, faster

Then:  streamlit run app.py
"""
import subprocess, sys, argparse, os

def step(title, cmd):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {title}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = "8000" if a.quick else "40000"
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
    step("1/4  Generating the mention dataset",
         ["data/generate_dataset.py", "--n", n, "--out", "data/mentions.csv"])
    step("2/4  Training and costing the spread model", ["src/train_spread_model.py"])
    step("3/4  Evaluating the classifier against human labels",
         ["src/evaluate_llm.py", "--n", "1200", "--engine", "heuristic"])
    step("4/4  Checking the policy index and guardrails", ["src/rag.py"])
    print("\nAll artefacts rebuilt. Now run:  streamlit run app.py")
