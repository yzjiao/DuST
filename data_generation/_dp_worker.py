#!/usr/bin/env python3
"""Data-parallel vLLM worker — launched by 1_generate.py as a subprocess."""

import argparse
import os
import pickle
import sys

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--is-thinking", action="store_true")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    with open(args.input_path, "rb") as f:
        messages = pickle.load(f)

    print(f"[DP worker GPU={gpu}] {len(messages)} prompts, TP={args.tp_size}")

    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        n=args.num_samples,
    )

    if args.is_thinking:
        outputs = llm.chat(
            messages, sampling_params, use_tqdm=True,
            chat_template_kwargs={"enable_thinking": True},
        )
    else:
        outputs = llm.chat(messages, sampling_params, use_tqdm=True)

    results = []
    for output in outputs:
        results.append((
            [o.text.strip() for o in output.outputs],
            [o.finish_reason for o in output.outputs],
        ))

    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[DP worker GPU={gpu}] Done, {len(results)} results saved")


if __name__ == "__main__":
    main()
