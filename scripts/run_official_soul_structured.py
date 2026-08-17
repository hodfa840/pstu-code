#!/usr/bin/env python3
"""Run official SOUL on the structured (templated) secret benchmark.

Mirrors run_official_soul_freeform.py but uses the 175 structured secrets and
evaluates with PSTU's exposure + WikiText-2 PPL at the main-table window
(max_length=256) so results are comparable to Tables 2-3.

Model is parametrized via env vars so the same script serves all model sizes:
  WAGLE_CLEAN_MODEL   HF name of the clean baseline (default EleutherAI/pythia-1.4b)
  WAGLE_INFECTED_PATH absolute path to the structured infected checkpoint
"""

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
SOUL_ROOT = REPO_ROOT / "external" / "recent-unlearning" / "SOUL"
CONFIG = SOUL_ROOT / "configs" / "structured_pii" / "SO_GradDiff.json"
RESULT_ROOT = SOUL_ROOT / "files" / "results" / "official_soul_structured"
EVAL_ROOT = PROJECT_ROOT / "results" / "official_soul_structured"
SECRETS = PROJECT_ROOT / "data" / "secrets_train.jsonl"

CLEAN_MODEL = os.environ.get("WAGLE_CLEAN_MODEL", "EleutherAI/pythia-1.4b")
INFECTED_PATH = os.environ.get(
    "WAGLE_INFECTED_PATH",
    str(REPO_ROOT / "models" / "pythia-1.4b-infected" / "final"),
)
PPL_ML = int(os.environ.get("PPL_MAX_LENGTH", "256"))

GRID = [
    {"lr": 5e-6, "gamma": 0.5, "rho": 0.04, "epochs": 1},
    {"lr": 5e-6, "gamma": 1.0, "rho": 0.04, "epochs": 1},
    {"lr": 1e-5, "gamma": 1.0, "rho": 0.04, "epochs": 1},
    {"lr": 1e-5, "gamma": 5.0, "rho": 0.04, "epochs": 1},
    {"lr": 5e-5, "gamma": 0.5, "rho": 0.04, "epochs": 1},
    {"lr": 5e-5, "gamma": 1.0, "rho": 0.04, "epochs": 1},
]


def run(cmd, cwd, env):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOUL_STRUCTURED_SECRETS"] = str(SECRETS)
    env.setdefault("HF_HOME", "/proj/berzelius-aiics-real/users/x_hodfa/huggingface_models")

    results = []
    for spec in GRID:
        name = (
            f"official_soul_gradiff_lr{spec['lr']}"
            f"_gamma{spec['gamma']}_rho{spec['rho']}"
        ).replace(".", "p").replace("-", "m")
        run_root = RESULT_ROOT / "runs" / name
        ckpt = run_root / "checkpoints"
        eval_path = EVAL_ROOT / name / "result.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)

        soul_cmd = [
            sys.executable,
            "src/exec/unlearn_model.py",
            "--config-file", CONFIG,
            "--overall.model_name", INFECTED_PATH,
            "--logger.name", name,
            "--unlearn.lr", str(spec["lr"]),
            "--unlearn.GA+FT.gamma", str(spec["gamma"]),
            "--unlearn.sophia_params.rho", str(spec["rho"]),
            "--unlearn.num_epochs", str(spec["epochs"]),
        ]
        run(soul_cmd, SOUL_ROOT, env)

        eval_cmd = [
            sys.executable,
            PROJECT_ROOT / "scripts" / "evaluate_model.py",
            "--model-path", ckpt,
            "--clean-model", CLEAN_MODEL,
            "--secrets-path", SECRETS,
            "--ppl-max-length", str(PPL_ML),
            "--output", eval_path,
        ]
        run(eval_cmd, PROJECT_ROOT, env)

        with open(eval_path) as f:
            result = json.load(f)
        result.update(spec)
        result["run_name"] = name
        result["checkpoint"] = str(ckpt)
        results.append(result)
        with open(EVAL_ROOT / "summary.json", "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(json.dumps({k: result[k] for k in ("run_name", "memorized", "ppl")},
                         indent=2), flush=True)

    ranked = sorted(results, key=lambda r: (r["memorized"], r["ppl"]))
    with open(EVAL_ROOT / "summary.json", "w") as f:
        json.dump({"results": results, "best_by_mem_then_ppl": ranked[:5]}, f, indent=2)
    print("Best official SOUL (structured) runs:", flush=True)
    print(json.dumps(ranked[:5], indent=2), flush=True)


if __name__ == "__main__":
    main()
