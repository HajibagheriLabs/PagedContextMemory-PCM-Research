"""Official prompts and scorers, loaded from the reference clones in third_party/ without editing them.

BABILong's prompts.py and metrics.py import nothing and are loaded as modules.
InfiniteBench's eval_utils.py and compute_scores.py import packages this project
does not install (jieba, rouge, evaluate) at module level, so the definitions used
here are taken from their source text: each named top-level function or
assignment is compiled verbatim into a private namespace. No line is rewritten.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from pcm.tasks.common import REPO_ROOT

THIRD_PARTY = REPO_ROOT / "third_party"
BABILONG_DIR = THIRD_PARTY / "babilong" / "babilong"
INFBENCH_DIR = THIRD_PARTY / "InfiniteBench" / "src"


# How each template set of InfiniteBench's MODEL_TO_PROMPT_TEMPLATE is recognised by content.
# "chat": the prompts for chat models, which end the passkey prompt with the question itself
# (no answer prefix such as "The pass key is"). Exactly one key of the official map has them.
TEMPLATE_FINGERPRINTS = {
    "chat": lambda t: t["passkey"].endswith("{context}\n\n{input}"),
}


def _template_keys(model_to_template: dict) -> dict:
    """Template-set name ("chat") -> the official key that selects it in create_msgs."""
    out = {}
    for name, test in TEMPLATE_FINGERPRINTS.items():
        keys = [k for k, t in model_to_template.items() if test(t)]
        if len(keys) != 1:
            raise LookupError(f"template set {name!r}: expected one matching key, found {len(keys)}")
        out[name] = keys[0]
    return out


def _load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"reference file missing: {path} (third_party/ is cloned by prompt A0)")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_definitions(path: Path, names: list[str], namespace: dict) -> dict:
    """Compile the named top-level functions and assignments of a source file into `namespace`."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            target = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            target = node.targets[0].id
        else:
            continue
        code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
        exec(code, namespace)
        found.add(target)
    missing = set(names) - found
    if missing:
        raise LookupError(f"{path.name}: definitions not found: {sorted(missing)}")
    return namespace


@lru_cache(maxsize=1)
def babilong() -> SimpleNamespace:
    prompts = _load_module("babilong_prompts", BABILONG_DIR / "prompts.py")
    metrics = _load_module("babilong_metrics", BABILONG_DIR / "metrics.py")
    return SimpleNamespace(
        DEFAULT_PROMPTS=prompts.DEFAULT_PROMPTS,
        DEFAULT_TEMPLATE=prompts.DEFAULT_TEMPLATE,
        get_formatted_input=prompts.get_formatted_input,
        compare_answers=metrics.compare_answers,
        preprocess_output=metrics.preprocess_output,
        TASK_LABELS=metrics.TASK_LABELS,
        source={"prompts": str(BABILONG_DIR / "prompts.py"), "metrics": str(BABILONG_DIR / "metrics.py")},
    )


INFBENCH_SCORERS = {
    "passkey": "get_score_one_passkey",
    "number_string": "get_score_one_number_string",
    "kv_retrieval": "get_score_one_kv_retrieval",
    "longbook_choice_eng": "get_score_one_longbook_choice_eng",
}


@lru_cache(maxsize=1)
def infbench() -> SimpleNamespace:
    prompt = _load_module("infbench_prompt", INFBENCH_DIR / "prompt.py")
    ns: dict = {"re": re, "string": string, "Counter": Counter, "Path": Path, "Optional": Optional}
    # The template dicts that MODEL_TO_PROMPT_TEMPLATE refers to, taken from prompt.py by their suffix.
    ns.update({k: v for k, v in vars(prompt).items() if k.endswith("_templates")})
    extract_definitions(
        INFBENCH_DIR / "eval_utils.py",
        ["DATA_NAME_TO_MAX_NEW_TOKENS", "MODEL_TO_PROMPT_TEMPLATE", "create_prompt", "get_answer", "create_msgs"],
        ns,
    )
    extract_definitions(INFBENCH_DIR / "compute_scores.py", ["first_int_match", *INFBENCH_SCORERS.values()], ns)
    return SimpleNamespace(
        create_prompt=ns["create_prompt"],
        get_answer=ns["get_answer"],
        max_new_tokens=ns["DATA_NAME_TO_MAX_NEW_TOKENS"],
        scorers={task: ns[fn] for task, fn in INFBENCH_SCORERS.items()},
        create_msgs=ns["create_msgs"],  # called with tokenizer=None: no truncation, returns (messages, prompt)
        template_keys=_template_keys(ns["MODEL_TO_PROMPT_TEMPLATE"]),
        source={"eval_utils": str(INFBENCH_DIR / "eval_utils.py"), "compute_scores": str(INFBENCH_DIR / "compute_scores.py")},
    )

