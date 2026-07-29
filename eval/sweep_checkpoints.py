#!/usr/bin/env python3
"""
Sweep eval across multiple checkpoint steps with multiple repeats.

For each checkpoint step, runs eval N times with different seeds and aggregates
results (mean +/- stderr).

Usage:
    # Eval all discovered checkpoint steps (LCB v6)
    python eval/sweep_checkpoints.py --config eval/configs/sweep_lcbv6.yaml

    # Eval specific steps
    python eval/sweep_checkpoints.py --config eval/configs/sweep_lcbv5.yaml --steps 100,200,300

    # Dry run (print commands without executing)
    python eval/sweep_checkpoints.py --config eval/configs/sweep_lcbv6.yaml --dry-run
"""

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml


def discover_steps(checkpoint_s3: str, save_freq: int) -> list[int]:
    """Auto-discover checkpoint steps from S3."""
    latest_file = "/tmp/latest_checkpointed_iteration.txt"
    s3_path = f"{checkpoint_s3}/latest_checkpointed_iteration.txt"

    subprocess.run(
        ["aws", "s3", "cp", s3_path, latest_file, "--quiet"],
        check=True,
    )
    with open(latest_file) as f:
        max_step = int(f.read().strip())

    steps = list(range(save_freq, max_step + 1, save_freq))
    print(f"Auto-discovered: max_step={max_step}, save_freq={save_freq} -> {len(steps)} steps")
    return steps


def run_single_eval(config, step, repeat_idx, project_root):
    """Run a single eval job (blocking)."""
    env = os.environ.copy()
    env.update({
        "S3_CKPT_DIR": f"{config['checkpoint_s3']}/global_step_{step}",
        "CKPT_STEP": str(step),
        "BASE_MODEL_S3": config.get("base_model_s3", ""),
        "BASE_MODEL_HF": config.get("base_model_hf", ""),
        "GENERATOR": config["generator"],
        "RANKER": config["ranker"],
        "REPEAT_IDX": str(repeat_idx),
        "N_ROLLOUTS": str(config["n_rollouts"]),
        "TP_SIZE": str(config["tp_size"]),
        "GPU_MEM": str(config["gpu_memory_utilization"]),
        "EVAL_TASK": config["eval_task"],
        "LCB_VERSION": config["lcb_version"],
        "SELECTION_MODE": config.get("selection_mode", "ranking"),
        "SAMPLING_PARAMS": config.get("sampling_params", ""),
        "RESULTS_DIR": config["results_dir"],
    })

    script = os.path.join(project_root, "eval", "eval_checkpoint.sh")
    result = subprocess.run(
        ["bash", script],
        env=env,
        capture_output=False,
    )
    return result.returncode == 0


def extract_metrics(results_dir, step, repeat_idx):
    """Extract metrics from eval results JSON."""
    pattern = os.path.join(results_dir, f"step_{step}", f"repeat_{repeat_idx}", "**", "*.json")
    files = glob.glob(pattern, recursive=True)

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, dict) and "results" in data:
                for task_name, task_data in data["results"].items():
                    if any(name in task_name for name in ["LiveCodeBench", "BigCodeBench", "CodeForces"]):
                        return task_data
            elif isinstance(data, dict) and any(k in data for k in ["accuracy", "pass_at_1", "ndcg"]):
                return data
        except Exception:
            continue
    return None


def aggregate_metrics(metrics_list):
    """Compute mean/stderr for numeric keys."""
    numeric_keys = set()
    for m in metrics_list:
        numeric_keys.update(k for k, v in m.items() if isinstance(v, (int, float)) and not k.startswith("_"))

    aggregated = {}
    for key in sorted(numeric_keys):
        values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            mean = float(np.mean(values))
            stderr = float(np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            aggregated[key] = {"mean": mean, "stderr": stderr, "n": len(values)}
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Sweep eval across checkpoint steps")
    parser.add_argument("--config", required=True, help="Sweep config YAML")
    parser.add_argument("--steps", help="Override: comma-separated step numbers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    project_root = str(Path(__file__).parent.parent)

    # Determine steps
    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    elif config.get("checkpoint_steps"):
        steps = config["checkpoint_steps"]
    else:
        steps = discover_steps(config["checkpoint_s3"], config["save_freq"])

    num_repeats = config["num_repeats"]
    total_jobs = len(steps) * num_repeats

    print(f"Sweep: {len(steps)} steps x {num_repeats} repeats = {total_jobs} eval jobs")
    print(f"  Steps: {steps}")
    print(f"  Config: {args.config}")

    if args.dry_run:
        for step in steps:
            for r in range(num_repeats):
                print(f"  [DRY RUN] step={step}, repeat={r}")
        return

    # Run all evals
    results = {}
    for step in steps:
        step_metrics = []
        for r in range(num_repeats):
            print(f"\n{'='*60}")
            print(f"Running: step={step}, repeat={r}/{num_repeats}")
            print(f"{'='*60}")

            success = run_single_eval(config, step, r, project_root)
            if not success:
                print(f"  FAILED: step={step}, repeat={r}")
                continue

            metrics = extract_metrics(config["results_dir"], step, r)
            if metrics:
                step_metrics.append(metrics)
                # Print key metric
                for key in ["accuracy", "pass_at_1"]:
                    if key in metrics:
                        print(f"  {key}: {metrics[key]:.4f}")
                        break

        if step_metrics:
            results[step] = aggregate_metrics(step_metrics)

    # Print summary table
    if not results:
        print("\nNo results collected!")
        return

    print(f"\n{'='*80}")
    print("Sweep Results (mean +/- stderr)")
    print(f"{'='*80}")

    first_agg = next(iter(results.values()))
    display_keys = [k for k in ["accuracy", "accuracy_easy", "accuracy_medium", "accuracy_hard", "pass_at_1"]
                    if k in first_agg]
    if not display_keys:
        display_keys = sorted(first_agg.keys())[:5]

    header = f"{'Step':>6}"
    for k in display_keys:
        header += f"  {k:>22}"
    header += f"  {'n':>3}"
    print(header)
    print("-" * len(header))

    for step in sorted(results.keys()):
        agg = results[step]
        row = f"{step:>6}"
        for k in display_keys:
            if k in agg:
                row += f"  {agg[k]['mean']:>8.4f} +/- {agg[k]['stderr']:.4f}"
            else:
                row += f"  {'N/A':>22}"
        row += f"  {agg[display_keys[0]]['n'] if display_keys else 0:>3}"
        print(row)

    print(f"{'='*80}")

    # Save JSON
    out_path = os.path.join(config["results_dir"], "sweep_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
