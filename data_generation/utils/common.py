"""Common utilities for data generation pipeline."""

import os
import glob
import json
from typing import Optional, List, Dict, Any

from jinja2 import Environment, StrictUndefined


class StatementAssembler:
    """Assembles problem statements using Jinja2 templates."""

    def __init__(self, template_str: str):
        self.env = Environment(
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.env.from_string(template_str)

    def assemble(self, components: Optional[dict] = None, **kwargs) -> str:
        if components is not None:
            return self.template.render(components=components, **kwargs)
        return self.template.render(**kwargs)


def load_template(template_name: str, base_dir: Optional[str] = None) -> str:
    """Load a Jinja2 template file."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
    template_path = os.path.join(base_dir, template_name)
    with open(template_path) as f:
        return f.read()


def estimate_token_count(text: str) -> int:
    """Estimate token count (~3 chars per token, conservative)."""
    if not text:
        return 0
    return len(text) // 3


def load_parquet_files(data_path: str, pattern: str = "*.parquet") -> List[Dict[str, Any]]:
    """Load parquet files from a path (single file or directory with glob pattern).

    Args:
        data_path: Path to a single parquet file, or directory containing parquet files
        pattern: Glob pattern when data_path is a directory

    Returns:
        List of examples (dicts)
    """
    import pyarrow.parquet as pq

    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted(glob.glob(os.path.join(data_path, pattern)))

    if not files:
        raise FileNotFoundError(f"No parquet files found at: {data_path}")

    print(f"Loading {len(files)} parquet file(s) from {data_path}")
    all_examples = []
    for pf_path in files:
        table = pq.read_table(pf_path)
        examples = table.to_pylist()
        all_examples.extend(examples)
        print(f"  Loaded {len(examples)} examples from {os.path.basename(pf_path)}")

    return all_examples
