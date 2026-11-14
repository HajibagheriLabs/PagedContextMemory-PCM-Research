# Paged Context Memory (PCM)

Research code for one question: can a small language model with a 32K native context
answer over 256K-token inputs, with no noticeable accuracy drop, by never looking at
more than its native window at once?

The long input is cut into **pages**. An **index** (embeddings, keyword scores and a
relational graph) records what each page contains and how pages relate. A **store**
keeps every page in compact form. For each question a **selector** picks the pages that
matter, an **expander** brings them back at full fidelity into the native window, and
the model answers. The design follows the way an operating system pages memory.

Reader: Qwen3-1.7B (Qwen3-0.6B for development and as a second scale point).
Hardware target: one 24 GB GPU.

**Status:** environment set up and checked (`reports/A0_setup.md`). No experiment has
been run yet. Nothing in this repository is a result until a report in `reports/` says so.

## Layout

```text
pcm/         library code
  index/     embeddings, BM25, graph, selection
  store/     KV cache adapter, codecs
  tasks/     synthetic tasks, BABILong, InfiniteBench, PG-19 streaming
  eval/      runner, metrics
  train/     adapters, router
configs/     one YAML per experiment; assets.yaml names models and datasets
scripts/     hw_probe, fetch_assets, check_config, smoke, ...
tests/       pytest
results/     <experiment>/runs.jsonl and env.json
reports/     one Markdown report per step, plus hardware and check JSON
paper/       claims ledger, draft, figures
DEVIATIONS.md   every departure from the specification, with its reason
```

## Setup

```text
uv venv .venv --python 3.11
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv -r requirements.lock.txt
python scripts/hw_probe.py
python scripts/fetch_assets.py
python scripts/check_config.py
python scripts/smoke.py
python -m pytest
```

`fetch_assets.py` uses the local Hugging Face cache first and downloads only what is
missing. Weights, datasets and KV stores are never committed.

## Rules the results follow

- Every number in a report comes from a file under `results/` or `reports/`.
- A failed run or a failed hypothesis is reported as failed.
- Nothing is tuned on test data.
- Fixed seeds, greedy decoding, thinking mode off.
