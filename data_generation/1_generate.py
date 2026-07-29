#!/usr/bin/env python3
"""
Self-Distillation Data Generation Pipeline.

Generates N solution samples per coding problem using vLLM, then verifies each
solution against test cases via sandbox execution.

Pipeline stages:
  Stage 0: Load raw data + format questions + generate prompts
  Stage 1: Generate self-distillation solutions (vLLM, multi-GPU data parallel)
  Stage 2: Sandbox verification
  Stage 3: Save parquet + upload to storage

Usage:
    # Single shard (for debugging)
    python data_generation/1_generate.py --shard-rank 0 --num-shards 1

    # Production: run 64 shards in parallel (e.g. via SLURM array)
    python data_generation/1_generate.py \
        --shard-rank $SLURM_ARRAY_TASK_ID \
        --num-shards 64 \
        --config configs/paths.yaml \
        --gen-config configs/data_gen.yaml

    # Without sandbox verification (generation only)
    python data_generation/1_generate.py --shard-rank 0 --num-shards 1 --skip-verification
"""

import argparse
import os
import sys
import time
import pickle
import subprocess
import tempfile

import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.common import (
    StatementAssembler,
    load_template,
    load_parquet_files,
    estimate_token_count,
)
from utils.model_utils import is_thinking_model
from utils.sandbox import SandboxClient
from utils.resource_manager import ensure_model_available, upload_file


def load_configs(args):
    with open(args.config) as f:
        paths_cfg = yaml.safe_load(f)
    with open(args.gen_config) as f:
        gen_cfg = yaml.safe_load(f)
    return paths_cfg, gen_cfg


def load_templates():
    """Load Jinja2 templates for prompt formatting."""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    statement_assembler = StatementAssembler(
        open(os.path.join(template_dir, "question.j2")).read()
    )
    stdin_assembler = StatementAssembler(
        open(os.path.join(template_dir, "prompt_stdin.j2")).read()
    )
    function_assembler = StatementAssembler(
        open(os.path.join(template_dir, "prompt_function.j2")).read()
    )
    return statement_assembler, stdin_assembler, function_assembler


def format_statement_to_question(statement, starter_code, assembler):
    if not statement or not isinstance(statement, dict):
        return ""
    return assembler.assemble(components=statement, starter_code=starter_code)


def parse_function_name(starter_code):
    """Parse entry function name from starter code."""
    import ast
    if not starter_code:
        return None
    try:
        code = starter_code
        lines = code.rstrip().split('\n')
        if lines and lines[-1].rstrip().endswith(':'):
            indent = len(lines[-1]) - len(lines[-1].lstrip()) + 4
            code = code + '\n' + ' ' * indent + 'pass'
        tree = ast.parse(code)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fn = node.name
        return fn
    except Exception:
        return None


def format_prompt(question, starter_code, problem_type, stdin_asm, function_asm, is_thinking):
    if problem_type == 'function':
        return function_asm.assemble(
            question=question, starter_code=starter_code, is_thinking_model=is_thinking
        )
    return stdin_asm.assemble(
        question=question, starter_code=starter_code, is_thinking_model=is_thinking
    )


def extract_python_code(text):
    """Extract Python code from markdown code blocks."""
    import re
    if not text:
        return None
    matches = re.findall(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    matches = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return None


def extract_test_cases(example):
    """Extract test cases from test_suites field."""
    test_suites = example.get('test_suites', [])
    if not isinstance(test_suites, list):
        return []
    test_cases = []
    for suite in test_suites:
        if not isinstance(suite, dict):
            continue
        for case in suite.get('cases', []):
            if isinstance(case, dict):
                inp = str(case.get('input', ''))
                out = str(case.get('output', ''))
                test_cases.append((inp, out))
    return test_cases


def run_dp_worker(gpu_ids, tp_size, model_name, gpu_mem, max_tokens,
                  temperature, top_k, top_p, rep_penalty, num_samples,
                  is_thinking, input_path, output_path):
    """Launch a data-parallel vLLM worker subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dp_worker.py")
    cmd = [
        sys.executable, worker_script,
        "--input-path", input_path,
        "--output-path", output_path,
        "--model-name", model_name,
        "--tp-size", str(tp_size),
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--repetition-penalty", str(rep_penalty),
        "--num-samples", str(num_samples),
    ]
    if is_thinking:
        cmd.append("--is-thinking")
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def generate_multi_gpu(valid_examples, model_name, tp_size, gpu_mem, max_tokens,
                       temperature, top_k, top_p, rep_penalty, num_samples,
                       is_thinking, shard_rank):
    """Generate using multiple GPUs with data parallelism."""
    import torch
    num_gpus = torch.cuda.device_count()
    num_workers = num_gpus // tp_size

    if num_workers <= 1 or not valid_examples:
        return None

    print(f"[Shard {shard_rank}] Multi-GPU DP: {num_workers} workers x TP={tp_size}")
    messages_all = [[{"role": "user", "content": ex['prompt']}] for ex in valid_examples]
    chunk_size = (len(messages_all) + num_workers - 1) // num_workers
    tmpdir = tempfile.mkdtemp(prefix="dp_gen_")

    procs = []
    for w in range(num_workers):
        chunk = messages_all[w * chunk_size : (w + 1) * chunk_size]
        if not chunk:
            continue
        gpu_start = w * tp_size
        gpu_ids = ",".join(str(gpu_start + g) for g in range(tp_size))
        input_path = os.path.join(tmpdir, f"input_{w}.pkl")
        output_path = os.path.join(tmpdir, f"output_{w}.pkl")
        with open(input_path, "wb") as f:
            pickle.dump(chunk, f)
        proc = run_dp_worker(gpu_ids, tp_size, model_name, gpu_mem, max_tokens,
                             temperature, top_k, top_p, rep_penalty, num_samples,
                             is_thinking, input_path, output_path)
        procs.append((w, proc, output_path))

    all_results = []
    failed = False
    for w, proc, output_path in procs:
        proc.wait()
        if proc.returncode != 0:
            print(f"[Shard {shard_rank}] DP worker {w} failed")
            failed = True
            continue
        with open(output_path, "rb") as f:
            all_results.extend(pickle.load(f))

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if failed:
        return None

    for ex, (texts, reasons) in zip(valid_examples, all_results):
        ex['self_distillation_outputs'] = texts
        ex['self_distillation_finish_reasons'] = reasons
    return True


def verify_solutions(examples, sandbox_client, max_tests, shard_rank):
    """Verify solutions against test cases using sandbox."""
    import asyncio

    print(f"[Shard {shard_rank}] Verifying {len(examples)} examples...")
    verified_count = 0

    async def verify_all():
        nonlocal verified_count
        semaphore = asyncio.Semaphore(128)

        async def verify_one(idx, ex):
            nonlocal verified_count
            outputs = ex.get('self_distillation_outputs', [])
            if not outputs:
                ex['self_distillation_pass_rates'] = []
                return

            test_cases = extract_test_cases(ex)
            if not test_cases:
                ex['self_distillation_pass_rates'] = [None] * len(outputs)
                return

            problem_type = ex.get('problem_type', 'stdin')
            function_name = None
            if problem_type == 'function':
                function_name = parse_function_name(ex.get('starter_code'))

            pass_rates = []
            for output in outputs:
                if not output or not output.strip():
                    pass_rates.append(None)
                    continue
                code = extract_python_code(output)
                if not code:
                    pass_rates.append(None)
                    continue
                async with semaphore:
                    pr = await sandbox_client.verify_async(
                        code, test_cases[:max_tests], function_name
                    )
                    pass_rates.append(pr)
                    if pr is not None:
                        verified_count += 1

            ex['self_distillation_pass_rates'] = pass_rates

        tasks = [verify_one(i, ex) for i, ex in enumerate(examples)]
        for i in range(0, len(tasks), 100):
            await asyncio.gather(*tasks[i:i+100])
            print(f"[Shard {shard_rank}] Verified {min(i+100, len(tasks))}/{len(tasks)}")

    asyncio.run(verify_all())
    total_samples = sum(len(ex.get('self_distillation_outputs', [])) for ex in examples)
    print(f"[Shard {shard_rank}] Verification: {verified_count}/{total_samples} samples verified")


def main():
    parser = argparse.ArgumentParser(description="Self-distillation data generation")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--gen-config", default="configs/data_gen.yaml")
    parser.add_argument("--shard-rank", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--sandbox-endpoints", default="sandbox_endpoints.txt",
                        help="File with sandbox endpoint URLs, one per line")
    args = parser.parse_args()

    paths_cfg, gen_cfg = load_configs(args)
    gen = gen_cfg["generation"]

    # Resolve model
    model_name = ensure_model_available(paths_cfg["model"])
    is_thinking = is_thinking_model(model_name)

    print(f"[Shard {args.shard_rank}] Model: {model_name}, Thinking: {is_thinking}")

    # Load templates
    statement_asm, stdin_asm, function_asm = load_templates()

    # ===== STAGE 0: Load and format data =====
    print(f"[Shard {args.shard_rank}] Stage 0: Loading data...")
    data_path = paths_cfg["dataset"]["raw_path"]
    all_examples = load_parquet_files(data_path)
    all_examples = all_examples[args.shard_rank::args.num_shards]
    if args.max_questions:
        all_examples = all_examples[:args.max_questions]
    print(f"[Shard {args.shard_rank}] {len(all_examples)} examples after sharding")

    for ex in all_examples:
        ex['question'] = format_statement_to_question(
            ex.get('statement', {}), ex.get('starter_code'), statement_asm
        )
        problem_type = ex.get('problem_type', 'stdin')
        starter_code = ex.get('starter_code')
        if problem_type == 'function':
            if not starter_code or not parse_function_name(starter_code):
                ex['problem_type'] = 'stdin'
        ex['prompt'] = format_prompt(
            ex['question'], ex.get('starter_code'), ex['problem_type'],
            stdin_asm, function_asm, is_thinking
        )

    # ===== STAGE 1: Generate solutions =====
    print(f"[Shard {args.shard_rank}] Stage 1: Generating {gen['num_samples']} samples/question...")
    max_prompt_tokens = int(gen['max_tokens'] * 0.8)
    valid_examples = []
    for ex in all_examples:
        if estimate_token_count(ex['prompt']) > max_prompt_tokens:
            ex['self_distillation_outputs'] = []
            ex['self_distillation_finish_reasons'] = []
        else:
            valid_examples.append(ex)

    dp_result = generate_multi_gpu(
        valid_examples, model_name, gen['tensor_parallel_size'],
        gen['gpu_memory_utilization'], gen['max_tokens'],
        gen['temperature'], gen['top_k'], gen['top_p'],
        gen['repetition_penalty'], gen['num_samples'],
        is_thinking, args.shard_rank,
    )

    if dp_result is None and valid_examples:
        # Fallback: single-process generation
        print(f"[Shard {args.shard_rank}] Falling back to single-process vLLM...")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=gen['tensor_parallel_size'],
            gpu_memory_utilization=gen['gpu_memory_utilization'],
            max_model_len=gen['max_tokens'],
            dtype="bfloat16",
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=gen['temperature'],
            top_k=gen['top_k'],
            top_p=gen['top_p'],
            repetition_penalty=gen['repetition_penalty'],
            max_tokens=gen['max_tokens'],
            skip_special_tokens=True,
            n=gen['num_samples'],
        )
        messages = [[{"role": "user", "content": ex['prompt']}] for ex in valid_examples]
        if is_thinking:
            outputs = llm.chat(messages, sampling_params, use_tqdm=True,
                               chat_template_kwargs={"enable_thinking": True})
        else:
            outputs = llm.chat(messages, sampling_params, use_tqdm=True)
        for ex, output in zip(valid_examples, outputs):
            ex['self_distillation_outputs'] = [o.text.strip() for o in output.outputs]
            ex['self_distillation_finish_reasons'] = [o.finish_reason for o in output.outputs]

    # ===== STAGE 2: Sandbox verification =====
    if not args.skip_verification:
        print(f"[Shard {args.shard_rank}] Stage 2: Sandbox verification...")
        endpoints = []
        if os.path.exists(args.sandbox_endpoints):
            with open(args.sandbox_endpoints) as f:
                endpoints = [line.strip() for line in f if line.strip()]
        if not endpoints:
            endpoints = [f"http://{paths_cfg['sandbox']['host']}:{paths_cfg['sandbox']['port']}"]
        sandbox_client = SandboxClient(
            endpoints=endpoints,
            timeout=paths_cfg['sandbox']['run_timeout'],
        )
        verify_solutions(
            all_examples, sandbox_client,
            paths_cfg['sandbox']['max_tests_per_example'],
            args.shard_rank,
        )
    else:
        print(f"[Shard {args.shard_rank}] Stage 2: Skipped (--skip-verification)")
        for ex in all_examples:
            n = len(ex.get('self_distillation_outputs', []))
            ex['self_distillation_pass_rates'] = [None] * n

    # ===== STAGE 3: Save =====
    print(f"[Shard {args.shard_rank}] Stage 3: Saving...")
    output_dir = paths_cfg["output"]["data_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for ex in all_examples:
        ex['generated_with'] = model_name

    parquet_filename = f"train-{args.shard_rank:03d}.parquet"
    local_path = os.path.join(output_dir, parquet_filename)
    table = pa.Table.from_pylist(all_examples)
    pq.write_table(table, local_path)
    print(f"[Shard {args.shard_rank}] Saved {len(all_examples)} examples to {local_path}")

    # Upload to S3 if configured
    data_s3 = paths_cfg["output"].get("data_s3")
    if data_s3:
        s3_path = f"{data_s3}/{parquet_filename}"
        upload_file(local_path, s3_path, paths_cfg.get("s3"))
        print(f"[Shard {args.shard_rank}] Uploaded to {s3_path}")

    print(f"[Shard {args.shard_rank}] Done!")


if __name__ == "__main__":
    main()
