import json
import sys
from gegenextract.optimization.utils import prompt_diff
from gegenextract.optimization.artifact import PromptArtifact


def main(a_path: str, b_path: str):
    a = PromptArtifact.model_validate_json(open(a_path).read()) if hasattr(PromptArtifact, 'model_validate_json') else PromptArtifact(**json.load(open(a_path)))
    b = PromptArtifact.model_validate_json(open(b_path).read()) if hasattr(PromptArtifact, 'model_validate_json') else PromptArtifact(**json.load(open(b_path)))
    print(prompt_diff(a, b))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_prompts.py promptA.json promptB.json")
    else:
        main(sys.argv[1], sys.argv[2])
