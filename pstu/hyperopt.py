"""Two-phase Optuna hyperparameter optimization for PSTU.

Phase 1: Multi-objective Pareto search (NSGA-II) minimizing exposure + PPL.
Phase 2: Single-objective refinement (TPE) minimizing a combined score.
"""

import gc
import json
import torch
import numpy as np
import optuna
import functools
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM

from pstu.method import apply_pstu
from pstu.evaluation import evaluate_perplexity, evaluate_exposure, load_secrets
from pstu.utils import detect_num_layers, GPUKeepAlive

print = functools.partial(print, flush=True)


class PSTUObjective:
    """Optuna objective for PSTU hyperparameter search."""

    def __init__(self, infected_state, clean_state, task_vectors_gpu,
                 saliency, tokenizer, secrets, clean_ppl,
                 n_layers, group_size, n_groups, clean_model_name,
                 device, trim=False, eval_device=None):
        self.infected_state = infected_state
        self.clean_state = clean_state
        self.task_vectors_gpu = task_vectors_gpu
        self.saliency = saliency
        self.tokenizer = tokenizer
        self.secrets = secrets
        self.clean_ppl = clean_ppl
        self.n_layers = n_layers
        self.group_size = group_size
        self.n_groups = n_groups
        self.clean_model_name = clean_model_name
        self.device = device
        self.eval_device = eval_device or device
        self.trim = trim

    def _suggest_params(self, trial):
        params = {}
        params["embed_alpha"] = trial.suggest_float("embed_alpha", 0.0, 4.0)
        params["head_alpha"] = trial.suggest_float("head_alpha", 0.0, 4.0)
        params["saliency_boost"] = trial.suggest_float("saliency_boost", 0.0, 2.0)
        for i in range(self.n_groups):
            params[f"g{i}_alpha"] = trial.suggest_float(f"g{i}_alpha", 0.0, 4.0)
        if self.trim:
            params["trim_fraction"] = trial.suggest_float("trim_fraction", 0.1, 0.99)
        return params

    def __call__(self, trial):
        params = self._suggest_params(trial)
        alphas = {"embed": params["embed_alpha"], "head": params["head_alpha"]}
        for i in range(self.n_groups):
            alphas[f"g{i}"] = params[f"g{i}_alpha"]
        trim_frac = params.get("trim_fraction", 0.0)

        try:
            new_state = apply_pstu(
                self.infected_state, self.clean_state, self.task_vectors_gpu,
                self.saliency, alphas, params["saliency_boost"],
                self.n_layers, self.group_size,
                trim_fraction=trim_frac, device=self.device,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.clean_model_name, torch_dtype=torch.float32,
                device_map={"": self.eval_device}, trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.load_state_dict(new_state, strict=False)
            del new_state
            torch.cuda.empty_cache()

            ppl = evaluate_perplexity(model, self.tokenizer)
            exp_result = evaluate_exposure(
                model, self.tokenizer, self.secrets, self.eval_device)
            exposure = exp_result["avg_exposure"]
            mem = exp_result["memorized"]
            total = exp_result["total_secrets"]
            del model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

        ppl_delta = (ppl - self.clean_ppl) / self.clean_ppl * 100
        trial.set_user_attr("memorized", mem)
        trial.set_user_attr("total", total)
        trial.set_user_attr("ppl_delta_pct", ppl_delta)

        return exposure, ppl


def run_hyperopt(model_name, infected_path, clean_model_name,
                 n_trials=500, timeout=36000, group_size=2,
                 trim=False, output_dir=None):
    """Run the full two-phase Optuna search.

    Parameters
    ----------
    model_name     : str  -- identifier (e.g. 'pythia-1.4b')
    infected_path  : str  -- path to infected model weights
    clean_model_name : str -- HuggingFace model name for clean baseline
    n_trials       : int  -- total trials (split 50/50 between phases)
    timeout        : int  -- max seconds per phase
    group_size     : int  -- transformer layers per alpha group
    trim           : bool -- enable PSTU-Trim
    output_dir     : str  -- where to save results
    """
    from pstu.method import compute_saliency

    keepalive = GPUKeepAlive()
    keepalive.start()

    output_dir = Path(output_dir or f"results/pstu_{'trim' if trim else 'comprehensive'}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    eval_device = f"cuda:{min(1, n_gpus - 1)}" if n_gpus > 1 else device

    secrets = load_secrets()
    print(f"Loaded {len(secrets)} secrets")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        clean_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Clean model PPL
    print("[1/5] Loading clean model...")
    dm = "auto" if n_gpus > 1 else {"": device}
    clean_model = AutoModelForCausalLM.from_pretrained(
        clean_model_name, torch_dtype=torch.float32,
        device_map=dm, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    clean_ppl = evaluate_perplexity(clean_model, tokenizer)
    clean_state = {k: v.cpu() for k, v in clean_model.state_dict().items()}
    del clean_model
    _clear_all_gpu_caches()
    print(f"  Clean PPL: {clean_ppl:.2f}")

    # Architecture detection
    n_layers = detect_num_layers(clean_state)
    n_groups = (n_layers + group_size - 1) // group_size
    print(f"  {n_layers} layers -> {n_groups} groups")

    # Saliency
    print("\n[2/5] Computing saliency...")
    sal_path = output_dir / f"{model_name}_saliency.json"
    if sal_path.exists():
        with open(sal_path) as f:
            saliency = json.load(f)
        print(f"  Loaded cached saliency ({len(saliency)} params)")
    else:
        saliency = compute_saliency(infected_path, clean_model_name, secrets, device)
        with open(sal_path, "w") as f:
            json.dump(saliency, f)
        print(f"  Computed saliency ({len(saliency)} params)")

    # Infected state + task vectors
    print("\n[3/5] Loading infected model and computing task vectors...")
    infected_model = AutoModelForCausalLM.from_pretrained(
        str(infected_path), torch_dtype=torch.float32,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    infected_state = {k: v.cpu() for k, v in infected_model.state_dict().items()}
    del infected_model
    _clear_all_gpu_caches()

    task_vectors_gpu = {}
    for name in infected_state:
        if name in clean_state:
            tv = (infected_state[name].float() - clean_state[name].float()).to(torch.bfloat16)
            if tv.abs().max().item() > 0:
                task_vectors_gpu[name] = tv.to(device)
    print(f"  {len(task_vectors_gpu)} task vectors on GPU")

    # Optuna Phase 1: Pareto
    half = n_trials // 2
    db_path = output_dir / f"{model_name}.db"
    storage = f"sqlite:///{db_path}"

    print(f"\n[4/5] Phase 1: Pareto ({half} trials)...")
    pareto_study = optuna.create_study(
        study_name=f"{model_name}_pareto", storage=storage,
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        load_if_exists=True,
    )
    obj = PSTUObjective(
        infected_state, clean_state, task_vectors_gpu, saliency,
        tokenizer, secrets, clean_ppl, n_layers, group_size,
        n_groups, clean_model_name, device, trim=trim,
        eval_device=eval_device,
    )
    existing = len(pareto_study.trials)
    remaining = max(0, half - existing)
    if remaining > 0:
        pareto_study.optimize(obj, n_trials=remaining, timeout=timeout)

    pareto_trials = _get_pareto_front(pareto_study, clean_ppl)

    # Phase 2: Refinement
    print(f"\n[5/5] Phase 2: Refinement ({half} trials)...")
    refined_study = optuna.create_study(
        study_name=f"{model_name}_refined", storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    for pt in pareto_trials:
        try:
            refined_study.enqueue_trial(pt["params"])
        except Exception:
            pass

    best_score = float("inf")
    best_result = None

    class RefinedObjective:
        def __call__(self_, trial):
            nonlocal best_score, best_result
            params = obj._suggest_params(trial)
            alphas = {"embed": params["embed_alpha"], "head": params["head_alpha"]}
            for i in range(n_groups):
                alphas[f"g{i}"] = params[f"g{i}_alpha"]
            trim_frac = params.get("trim_fraction", 0.0)

            try:
                new_state = apply_pstu(
                    infected_state, clean_state, task_vectors_gpu,
                    saliency, alphas, params["saliency_boost"],
                    n_layers, group_size,
                    trim_fraction=trim_frac, device=device,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    clean_model_name, torch_dtype=torch.float32,
                    device_map={"": eval_device}, trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.load_state_dict(new_state, strict=False)
                del new_state
                torch.cuda.empty_cache()

                ppl = evaluate_perplexity(model, tokenizer)
                exp_result = evaluate_exposure(
                    model, tokenizer, secrets, eval_device)
                exposure = exp_result["avg_exposure"]
                mem = exp_result["memorized"]
                del model
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()

            ppl_delta = (ppl - clean_ppl) / clean_ppl * 100
            score = exposure + (ppl - clean_ppl) / 20.0

            trial.set_user_attr("exposure", exposure)
            trial.set_user_attr("memorized", mem)
            trial.set_user_attr("ppl", ppl)
            trial.set_user_attr("ppl_delta_pct", ppl_delta)

            if score < best_score:
                best_score = score
                best_result = {
                    "params": params, "exposure": exposure,
                    "memorized": mem, "ppl": ppl,
                    "ppl_delta_pct": ppl_delta, "score": score,
                }
                print(f"  >>> New best! Trial {trial.number}: "
                      f"Exp={exposure:.4f} PPL={ppl:.2f} Score={score:.4f}")
            return score

    existing_ref = len(refined_study.trials)
    remaining_ref = max(0, half - existing_ref)
    if remaining_ref > 0:
        refined_study.optimize(RefinedObjective(), n_trials=remaining_ref,
                               timeout=timeout)

    # Save final results
    final = {
        "model": model_name, "clean_ppl": clean_ppl,
        "n_layers": n_layers, "group_size": group_size,
        "total_trials": len(pareto_study.trials) + len(refined_study.trials),
        "best_refined": best_result, "pareto_front": pareto_trials,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / f"{model_name}_final.json", "w") as f:
        json.dump(final, f, indent=2)

    keepalive.stop()
    print(f"\nDone: {datetime.now()}")
    return best_result


def _clear_all_gpu_caches():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    gc.collect()


def _get_pareto_front(study, clean_ppl):
    trials = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        exposure, ppl = t.values
        trials.append({
            "number": t.number,
            "exposure": exposure, "ppl": ppl,
            "ppl_delta": (ppl - clean_ppl) / clean_ppl * 100,
            "mem": t.user_attrs.get("memorized", -1),
            "total": t.user_attrs.get("total", 175),
            "params": t.params,
        })
    trials.sort(key=lambda t: t["exposure"])
    return trials
