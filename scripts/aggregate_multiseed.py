#!/usr/bin/env python3
"""Aggregate results from multiple PSTU seeds and compute statistics."""

import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="Directories containing per-seed results")
    parser.add_argument("--output", required=True,
                        help="Output JSON file for aggregated results")
    args = parser.parse_args()

    results = []
    for d in args.results_dirs:
        p = Path(d)
        # Find the *_final.json file
        final_files = list(p.glob("*_final.json"))
        if not final_files:
            print(f"Warning: No *_final.json found in {d}")
            continue
        with open(final_files[0]) as f:
            data = json.load(f)
            if data.get("best_refined"):
                results.append({
                    "seed": data.get("seed", "unknown"),
                    "memorized": data["best_refined"]["memorized"],
                    "ppl": data["best_refined"]["ppl"],
                    "ppl_delta_pct": data["best_refined"]["ppl_delta_pct"],
                    "exposure": data["best_refined"]["exposure"],
                    "clean_ppl": data.get("clean_ppl"),
                })

    if not results:
        print("No results found!")
        return

    # Compute statistics
    mem_vals = [r["memorized"] for r in results]
    ppl_vals = [r["ppl"] for r in results]
    delta_vals = [r["ppl_delta_pct"] for r in results]
    exp_vals = [r["exposure"] for r in results]

    summary = {
        "n_seeds": len(results),
        "individual_results": results,
        "statistics": {
            "memorized": {
                "mean": float(np.mean(mem_vals)),
                "std": float(np.std(mem_vals)),
                "min": int(np.min(mem_vals)),
                "max": int(np.max(mem_vals)),
            },
            "ppl": {
                "mean": float(np.mean(ppl_vals)),
                "std": float(np.std(ppl_vals)),
                "min": float(np.min(ppl_vals)),
                "max": float(np.max(ppl_vals)),
            },
            "ppl_delta_pct": {
                "mean": float(np.mean(delta_vals)),
                "std": float(np.std(delta_vals)),
                "min": float(np.min(delta_vals)),
                "max": float(np.max(delta_vals)),
            },
            "exposure": {
                "mean": float(np.mean(exp_vals)),
                "std": float(np.std(exp_vals)),
                "min": float(np.min(exp_vals)),
                "max": float(np.max(exp_vals)),
            },
        }
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Multi-Seed Summary ({len(results)} seeds) ===")
    print(f"Memorized: {summary['statistics']['memorized']['mean']:.1f} ± {summary['statistics']['memorized']['std']:.1f} "
          f"(range: {summary['statistics']['memorized']['min']}-{summary['statistics']['memorized']['max']})")
    print(f"PPL:       {summary['statistics']['ppl']['mean']:.2f} ± {summary['statistics']['ppl']['std']:.2f} "
          f"(range: {summary['statistics']['ppl']['min']:.2f}-{summary['statistics']['ppl']['max']:.2f})")
    print(f"ΔPPL%:     {summary['statistics']['ppl_delta_pct']['mean']:.2f} ± {summary['statistics']['ppl_delta_pct']['std']:.2f}")
    print(f"Exposure:  {summary['statistics']['exposure']['mean']:.4f} ± {summary['statistics']['exposure']['std']:.4f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
