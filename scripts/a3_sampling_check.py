"""Why the model card's sampling settings can coincide with greedy decoding (A3, E0 sampling check).

    python scripts/a3_sampling_check.py

For every example of the 50-example sampling subset (configs/a3_floor.yaml), follow the
greedy answer of Qwen3-1.7B token by token and record, at each step, the probability of
the chosen token after temperature 0.7 over the full vocabulary. With top-p 0.8, a step
whose top token has probability >= 0.8 leaves only that token in the nucleus, so a
sampled answer cannot differ from the greedy one there. (Top-k 20, applied first,
renormalises over 20 tokens and can only raise that probability: the check is conservative.)

Writes results/A3_floor/sampling_confidence.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from pcm.env import collect_env  # noqa: E402
from pcm.eval import runner as R  # noqa: E402
from pcm.health import preflight  # noqa: E402
from pcm.prefill import _as_row, _forward_blocks, _library_cache  # noqa: E402
from pcm.runlog import atomic_write_json  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402


def main() -> int:
    tcfg = load_task_config()
    acfg = yaml.safe_load((REPO_ROOT / "configs" / "a3_floor.yaml").read_text(encoding="utf-8"))
    entry = next(e for e in acfg["experiments"]["E0_floor"] if e["method"] == "sampled-1.7B")
    sp = acfg["methods"]["sampled-1.7B"]["sampling"]
    preflight(min_free_vram_gib=8.0)
    m = R.make_method(acfg["methods"]["native-1.7B"])
    m.setup()
    out = []
    for task in acfg["task_sets"][entry["tasks"]]:
        for L in entry["lengths"]:
            exs = R.load_examples(task, L, max(entry["indices"]) + 1, acfg["seed"], tcfg)
            for ex in [e for e in exs if e["index"] in entry["indices"]]:
                prep = m.prepare(ex, R.prompt_for(ex, tcfg))
                ids = _as_row(prep["ids"], m.model.device)
                probs = []
                with torch.no_grad():
                    lib = _library_cache(m.model, None, ids.shape[1] + ex["max_new_tokens"])
                    logits = _forward_blocks(m.model, ids, lib, 2048, 0, True)
                    for _ in range(int(ex["max_new_tokens"])):
                        p = torch.softmax(logits.float() / sp["temperature"], -1)
                        top = int(p.argmax())
                        probs.append(float(p[top]))
                        if top in m.stops:
                            break
                        logits = _forward_blocks(m.model, torch.tensor([[top]], device=ids.device), lib, 1, 0, True)
                out.append({"task": task, "example_id": ex["id"], "steps": len(probs), "min_top_prob": min(probs),
                            "steps_below_top_p": sum(x < sp["top_p"] for x in probs)})
    env = collect_env()
    res = {
        "code": {"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]},
        "sampling": sp, "examples": len(out),
        "examples_with_every_step_at_or_above_top_p": sum(o["steps_below_top_p"] == 0 for o in out),
        "min_top_prob_over_all": min(o["min_top_prob"] for o in out),
        "rows": out,
    }
    atomic_write_json(REPO_ROOT / "results" / "A3_floor" / "sampling_confidence.json", res)
    print({k: v for k, v in res.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
