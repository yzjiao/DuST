#!/usr/bin/env python3
"""
Process self-distillation data for code selector training.

Reads parquet files from the generation output, filters and groups candidates:
1. Filter queries where all N candidates pass or all fail
2. Split candidates into groups of K (default 4)
3. Keep only groups with at least one pass and one fail
4. Save to processed_train_nK.parquet

Usage:
    python data_generation/2_process_data.py --config configs/paths.yaml
    python data_generation/2_process_data.py --input-dir ./output/sd_data --group-size 4
"""

import argparse
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def extract_python_code(text):
    """Extract Python code from markdown code blocks."""
    if not text:
        return text
    matches = re.findall(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    matches = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return text


def process_parquet(local_path, group_size=4, max_pass_ratio=1.0):
    """Process a single parquet file into candidate groups."""
    pf = pq.ParquetFile(local_path)
    schema_names = pf.schema_arrow.names

    required = {'prompt', 'self_distillation_outputs', 'self_distillation_pass_rates'}
    if not required.issubset(set(schema_names)):
        missing = required - set(schema_names)
        print(f"  WARNING: missing columns {missing}, skipping")
        return [], {'total_queries': 0, 'filtered_all_correct': 0, 'filtered_all_wrong': 0}

    cols = ['prompt', 'self_distillation_outputs', 'self_distillation_pass_rates']
    if 'canonical_id' in schema_names:
        cols.append('canonical_id')

    rows = []
    total_queries = 0
    filtered_all_correct = 0
    filtered_all_wrong = 0
    filtered_high_pass_ratio = 0

    for batch in pf.read(columns=cols).to_batches():
        has_canonical = 'canonical_id' in batch.schema.names
        for i in range(len(batch)):
            prompt = batch.column('prompt')[i].as_py()
            outputs = batch.column('self_distillation_outputs')[i].as_py()
            pass_rates = batch.column('self_distillation_pass_rates')[i].as_py()
            canonical_id = batch.column('canonical_id')[i].as_py() if has_canonical else None

            if not outputs or not pass_rates:
                continue

            valid = [(o, pr) for o, pr in zip(outputs, pass_rates) if pr is not None]
            if not valid:
                continue

            valid_outputs, valid_pass_rates = zip(*valid)
            valid_outputs = list(valid_outputs)
            valid_pass_rates = list(valid_pass_rates)
            total_queries += 1

            if all(pr == 1.0 for pr in valid_pass_rates):
                filtered_all_correct += 1
                continue
            if all(pr == 0.0 for pr in valid_pass_rates):
                filtered_all_wrong += 1
                continue

            pass_ratio = sum(1 for pr in valid_pass_rates if pr == 1.0) / len(valid_pass_rates)
            if pass_ratio > max_pass_ratio:
                filtered_high_pass_ratio += 1
                continue

            n = len(valid_outputs)
            for g in range(n // group_size):
                start = g * group_size
                end = start + group_size
                group_outputs = valid_outputs[start:end]
                group_pass_rates = valid_pass_rates[start:end]

                has_correct = any(pr == 1.0 for pr in group_pass_rates)
                has_incorrect = any(pr < 1.0 for pr in group_pass_rates)

                if has_correct and has_incorrect:
                    rows.append({
                        'canonical_id': canonical_id,
                        'prompt': prompt,
                        'candidates': group_outputs,
                        'candidates_code': [extract_python_code(o) for o in group_outputs],
                        'pass_rates': group_pass_rates,
                    })

    stats = {
        'total_queries': total_queries,
        'filtered_all_correct': filtered_all_correct,
        'filtered_all_wrong': filtered_all_wrong,
        'filtered_high_pass_ratio': filtered_high_pass_ratio,
    }
    return rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/paths.yaml', help='Paths config')
    parser.add_argument('--gen-config', default='configs/data_gen.yaml', help='Generation config')
    parser.add_argument('--input-dir', help='Override input directory (default from config)')
    parser.add_argument('--output-dir', help='Override output directory (default from config)')
    parser.add_argument('--group-size', type=int, help='Override group size (default from gen config)')
    parser.add_argument('--max-pass-ratio', type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    with open(args.gen_config) as f:
        gen_cfg = yaml.safe_load(f)

    input_dir = args.input_dir or paths_cfg["output"]["data_dir"]
    output_dir = args.output_dir or paths_cfg["output"]["processed_data_dir"]
    group_size = args.group_size or gen_cfg["processing"]["group_size"]
    max_pass_ratio = args.max_pass_ratio

    # Find parquet files
    import glob
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "train-*.parquet")))
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")

    if not parquet_files:
        print("No parquet files found, exiting.")
        sys.exit(1)

    all_rows = []
    total_queries = 0
    filtered_all_correct = 0
    filtered_all_wrong = 0
    filtered_high_pass_ratio = 0

    for i, filepath in enumerate(parquet_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(parquet_files)}] Processing {filename}...")
        rows, stats = process_parquet(filepath, group_size=group_size, max_pass_ratio=max_pass_ratio)
        all_rows.extend(rows)
        total_queries += stats['total_queries']
        filtered_all_correct += stats['filtered_all_correct']
        filtered_all_wrong += stats['filtered_all_wrong']
        filtered_high_pass_ratio += stats.get('filtered_high_pass_ratio', 0)
        print(f"  -> {len(rows)} groups, total so far: {len(all_rows)}")

    if not all_rows:
        print("No valid data, exiting.")
        sys.exit(1)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Processing Statistics")
    print(f"{'='*60}")
    print(f"Total queries:              {total_queries}")
    print(f"Filtered (all correct):     {filtered_all_correct} ({100*filtered_all_correct/max(total_queries,1):.1f}%)")
    print(f"Filtered (all wrong):       {filtered_all_wrong} ({100*filtered_all_wrong/max(total_queries,1):.1f}%)")
    remaining = total_queries - filtered_all_correct - filtered_all_wrong - filtered_high_pass_ratio
    print(f"Remaining queries:          {remaining} ({100*remaining/max(total_queries,1):.1f}%)")
    print(f"Total groups of {group_size}:         {len(all_rows)}")
    print(f"Avg groups per query:       {len(all_rows)/max(remaining,1):.2f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_train_n{group_size}.parquet")
    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, output_path)
    print(f"\nSaved {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
