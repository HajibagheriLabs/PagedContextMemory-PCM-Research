"""A8 (E8): the cells of scripts/run_a8.py, and KV mode with the int4 store codec end to end.

CPU tests check that E8 runs the method configurations exactly as fixed in A4, A5, A6 and A7 (same hashes, so answers of
identical prompts are reusable and earlier rows are the same method), and the cell ranges. The GPU test runs PCMKV with the
int4 codec through `execute` on one 8K synthetic example with the development reader, against the bf16 store.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run_a8():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("run_a8", ROOT / "scripts" / "run_a8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_e8_methods_are_the_fixed_configurations():
    from pcm.tasks.common import config_hash

    mod = _run_a8()
    cfg = mod.load_config()
    m = cfg["_methods"]
    assert config_hash(m["P-T"]) == "a9f9fa050cff"          # A5's rows of pcm-p512-b8192 (configs/pcm_text.yaml)
    assert config_hash(m["M2"]) == "a6d04f3a2c47"           # A4's rows of m2-hybrid-1.7B
    kv = m["P-KV"]
    assert kv["method"] == "pcm_kv" and kv["kv"]["codec"] == "int4"
    assert (kv["kv"]["temperature"], kv["kv"]["scale"]) == (1.0, 0.9)
    same = {k: v for k, v in kv.items() if k not in ("method", "name", "kv")}
    assert same == {k: v for k, v in m["P-T"].items() if k not in ("method", "name")}   # the same pages as P-T


def test_e8_cells():
    mod = _run_a8()
    cfg = mod.load_config()
    cells = mod.e8_cells(cfg)
    kv = [c for c in cells if c["label"] == "P-KV"]
    assert len(kv) == 13 * 4 + 4                              # every retained task and length, and InfiniteBench
    assert sum(c["n_run"] for c in kv) == 13 * 4 * 100 + 4 * 100 + 3 * 200
    synth256 = [c for c in cells if c["task"].startswith("synth:") and c["length"] == "256k"]
    assert {(c["label"], c["lo"], c["hi"]) for c in synth256} == {("P-KV", 0, 300), ("P-T", 0, 100), ("P-T", 100, 300),
                                                                  ("M2", 0, 100), ("M2", 100, 300)}
    assert all(c["answer_from"] for c in synth256 if c["lo"] == 0 and c["hi"] == 100)
    assert not any(c["answer_from"] for c in kv)
    # rows of a ranged cell are counted by their example index only
    c = next(x for x in synth256 if x["label"] == "P-T" and x["lo"] == 100)
    from pcm.tasks.common import config_hash

    key = (config_hash(c["mcfg"]), c["task"], c["length"])
    by = {key: [{"index": i, "status": "ok", "timings": {"total_s": 1.0}} for i in (5, 99, 100, 150, 299)]}
    assert mod.cell_stats(c, by)["done"] == 3


@pytest.mark.gpu
def test_pcm_kv_with_the_int4_store_end_to_end(request, tmp_path):
    from pcm.health import EXIT_GPU_LOST, gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    st = gpu_status()
    if not st["ok"]:
        pytest.exit(f"no usable GPU: {st['reason']}", returncode=EXIT_GPU_LOST)
    from pcm.eval import runner as R
    from pcm.tasks.common import load_task_config

    mod = _run_a8()
    cfg = mod.load_config()
    tcfg = load_task_config()
    ex = R.load_examples("synth:S1", "8k", 1, 0, tcfg)[0]
    out = {}
    for codec in ("bf16", "int4"):
        mcfg = {**cfg["_methods"]["P-KV"], "reader": "reader_dev", "index_cache": str(tmp_path / "index")}
        mcfg["kv"] = {**mcfg["kv"], "codec": codec}
        m = R.make_method(mcfg)
        m.setup()
        m.answer_cache = {}
        res = m.execute(m.prepare(ex, R.prompt_for(ex, tcfg)))
        out[codec] = res
    a, b = out["bf16"], out["int4"]
    assert a["selection"]["units_b64"] == b["selection"]["units_b64"]          # the codec does not touch the selection
    assert a["selection"]["text_prompt_sha"] == b["selection"]["text_prompt_sha"]
    assert a["prompt_sha"] != b["prompt_sha"]                                  # nor is an int4 answer reused for bf16
    assert b["selection"]["codec"] == "int4" and "store_codec_bytes" in b["timings"]
    per_token = b["timings"]["store_codec_bytes"] / b["timings"]["encoded_tokens"]
    assert 28_000 < per_token < 34_000                                         # 4 bits + float16 parameters, vs 114,688 in bf16
    assert b["prediction"]
