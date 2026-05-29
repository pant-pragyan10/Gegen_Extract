import json
import sys


def inspect(path: str):
    with open(path) as f:
        data = json.load(f)
    for gen in data.get("generations", []):
        print(f"Gen {gen['generation']}: best score {max((c.get('score',0) for c in gen['candidates']), default=0)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "opt_run_sample.json"
    inspect(path)
