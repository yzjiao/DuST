"""
Prepare code ranking data for verl GRPO training.

Converts processed self-distillation data (from 2_process_data.py) into the format
expected by verl's RLHFDataset: prompt (chat messages) + ground_truth (pass_rates).

Usage:
    python training/prepare_data.py --config configs/paths.yaml
"""

import argparse
import json
import os
import random

import datasets
import pyarrow.parquet as pq
import yaml


RANKING_PROMPT_TEMPLATE = """You will be given a question (problem specification) and {n_candidates} candidate solutions. You will analyze these candidates and RANK them from most likely to least likely to pass all tests. You will NOT return anything except for the ranking.

Question: {problem_description}

{candidate_solutions}

Analyze each candidate solution, analyze all the strengths and weaknesses, and rank them from most effective to least effective. Output ONLY the ranking as comma-separated numbers from best to worst (e.g., "2, 4, 1, 3").

Ranking (best to worst): """


def format_candidates(solutions: list[str]) -> str:
    parts = []
    for i, code in enumerate(solutions):
        parts.append(f"--- CANDIDATE {i+1} ---\n```python\n{code}\n```\n")
    return "\n".join(parts)


def build_prompt(query: str, candidate_solutions: list[str]) -> str:
    return RANKING_PROMPT_TEMPLATE.format(
        n_candidates=len(candidate_solutions),
        problem_description=query,
        candidate_solutions=format_candidates(candidate_solutions),
    )


def load_parquet(path: str) -> list:
    """Load parquet file and return list of dicts."""
    pf = pq.ParquetFile(path)
    rows = []
    for batch in pf.iter_batches(columns=['prompt', 'candidates_code', 'pass_rates']):
        for i in range(len(batch)):
            candidates = batch.column('candidates_code')[i].as_py()
            pass_rates = batch.column('pass_rates')[i].as_py()
            if isinstance(candidates, str):
                candidates = json.loads(candidates)
            if isinstance(pass_rates, str):
                pass_rates = json.loads(pass_rates)
            rows.append({
                'prompt': batch.column('prompt')[i].as_py(),
                'candidates': candidates,
                'pass_rates': pass_rates,
            })
    return rows


def build_records(raw_data: list) -> list:
    records = []
    for idx, item in enumerate(raw_data):
        query = item["prompt"]
        candidates = item["candidates"]
        pass_rates = item["pass_rates"]

        assert len(candidates) == len(pass_rates), (
            f"Sample {idx}: length mismatch ({len(candidates)} vs {len(pass_rates)})"
        )

        prompt_text = build_prompt(query, candidates)
        record = {
            "data_source": "code_ranking",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "code_ranking",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(pass_rates),
            },
            "extra_info": {
                "index": idx,
                "n_candidates": len(candidates),
            },
        }
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--input-path", help="Override: path to processed parquet")
    parser.add_argument("--output-dir", help="Override: output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)

    processed_dir = paths_cfg["output"]["processed_data_dir"]
    input_path = args.input_path or os.path.join(processed_dir, "processed_train_n4.parquet")
    output_dir = args.output_dir or processed_dir

    print(f"Loading data from {input_path}...")
    raw_data = load_parquet(input_path)
    print(f"Loaded {len(raw_data)} samples")

    all_records = build_records(raw_data)

    # Split train/val
    random.seed(args.seed)
    random.shuffle(all_records)
    split_idx = int(len(all_records) * (1 - args.val_ratio))
    train_records = all_records[:split_idx]
    test_records = all_records[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_records)
    test_ds = datasets.Dataset.from_list(test_records)

    train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(output_dir, "test.parquet"))

    print(f"Train: {len(train_ds)} samples")
    print(f"Test:  {len(test_ds)} samples")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
