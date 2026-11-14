# A0 — Setup report

Date: 2026-11-14. Machine: local workstation (Phase A).

This step builds the environment and checks that the assumptions behind the memory
tables of the specification hold on the real model files. **No experiment was run.**
Every number below is read from a file written by a script in `scripts/`:

| File | Written by |
|---|---|
| `reports/hw_local.json` | `scripts/hw_probe.py` |
| `reports/assets.json` | `scripts/fetch_assets.py --offline` |
| `reports/config_check.json` | `scripts/check_config.py` |
| `reports/smoke.json` | `scripts/smoke.py` |
| `requirements.lock.txt` | `uv pip freeze` |

## 1. Outcome

| Check | Result |
|---|---|
| Environment created, packages installed, lock file written | done (82 packages) |
| GPU visible to PyTorch, bf16 supported | yes |
| Models and datasets present and complete | yes, all from the local cache, nothing downloaded |
| Config assertions (28 layers, 16 heads, 8 KV heads, head dimension 128) | passed for both readers |
| KV cache per token equals 114,688 bytes | yes for both readers |
| `max_position_embeddings` | 40,960 for both readers, as the specification expects |
| Smoke test (reader answers, embedder embeds) | passed |
| Unit tests | 8 passed |

Nothing contradicts the specification. Eight deviations are recorded (section 7); none
changes a result.

## 2. Hardware (`reports/hw_local.json`)

| Item | Value |
|---|---|
| OS | Windows 10, build 10.0.19044, AMD64 |
| CPU | Intel Core i7-4790K @ 4.00 GHz, 4 cores / 8 threads |
| RAM | 15.85 GiB total, 8.74 GiB free at probe time |
| Page file | 13.0 GiB at probe time |
| GPU | NVIDIA GeForce RTX 3090, 24.0 GiB VRAM, 22.79 GiB free at probe time |
| NVIDIA driver | 610.88 |
| CUDA runtime (PyTorch build) | 12.6, cuDNN 9.10.2 |
| Disk, project drive and HF cache drive (both `C:`) | 465.11 GiB total, 144.65 GiB free |

Notes:

- RAM is the tight resource, as the specification says: only 8.74 GiB was free with the
  desktop running. Close other programs before long runs.
- The page file was 13.0 GiB at probe time. The specification asks that it be able to
  grow past 32 GB (system-managed on the NVMe); this was not checked from the script.
- `nvidia-smi` showed 579 MiB of VRAM in use before anything was loaded; with a CUDA
  context open, PyTorch reported 22.79 GiB free of 24.0.

## 3. Software (`requirements.lock.txt`)

| Package | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.14.1+cu126 |
| transformers | 5.19.0 |
| accelerate | 1.14.0 |
| datasets | 5.0.0 |
| huggingface_hub | 1.33.0 |
| sentence-transformers | 5.6.0 |
| peft | 0.19.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.23.3 |
| numpy / scipy / pandas | 2.4.6 / 1.17.1 / 3.0.6 |
| pyarrow | 26.0.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| zstandard / lz4 / pyppmd | 0.25.0 / 4.4.5 / 1.3.1 |
| rank-bm25 | 0.2.2 |
| pytest | 9.1.1 |
| pyyaml / tqdm / psutil | 6.0.3 / 4.70.1 / 7.2.2 |

`pyppmd` (optional) installed without trouble. `flash-attn`, `vllm`, `bitsandbytes` and
`nvcomp` are not installed.

## 4. Models and data (`reports/assets.json`)

All assets were already in the Hugging Face cache (`C:\Users\Hadi\hf\hub`).
`scripts/fetch_assets.py` resolves each one from the cache, checks that the required
files exist and are non-empty, and reads a first record from every dataset. It downloads
only what is missing; here that was nothing. Sizes are bytes on disk.

| Asset | Size on disk | Revision |
|---|---|---|
| Qwen/Qwen3-1.7B (reader) | 3.80 GiB | `70d244cc86cc` |
| Qwen/Qwen3-1.7B-Base | 3.22 GiB | `ea980cb0a6c2` |
| Qwen/Qwen3-0.6B (development reader) | 1.41 GiB | `c1899de289a0` |
| Qwen/Qwen3-Embedding-0.6B | 1.12 GiB | `97b0c614be4d` |
| Qwen/Qwen3-Reranker-0.6B (optional) | 1.12 GiB | `e61197ed4502` |
| RMT-team/babilong | 26.29 GiB (1.91 GiB for qa1–qa10 at 0k–256k) | `ee0d588794c7` |
| RMT-team/babilong-train-5k-samples | 23.53 GiB | `b3513ef7c25c` |
| xinrongzhang2022/InfiniteBench | 2.33 GiB (806.4 MiB for the four tasks used) | `90f039433361` |
| emozilla/pg19 (validation and test only) | 34.0 MiB | `c021754c8e01` |

Qwen/Qwen3-4B is not on this machine; it belongs to Phase B.

What the dataset cards and files say (checked, not assumed):

- **BABILong.** Lengths are *configs* and tasks are *splits*. Files are
  `data/<task>/<length>.json`, each a list of 100 rows with fields `input`, `question`,
  `target`. All ten tasks qa1–qa10 are present at all ten lengths 0k–256k. The 512k, 1M
  and 10M files are also on disk (Phase B uses 512k and 1M). Size per length for the ten
  tasks: 0.4 MiB (0k), 2.8, 6.6, 14.2, 29.6, 60.1, 122.0, 245.5, 493.6, 984.6 MiB (256k).
- **BABILong train.** Tasks qa1–qa10 at lengths 0k to 128k. Used for adapters only.
- **InfiniteBench.** One jsonl file per task, read with the Features spec from the
  dataset card (`id`, `context`, `input`, `answer`, `options`). Rows: passkey 590,
  number_string 590, kv_retrieval 500, longbook_choice_eng 229. First-row context
  lengths range from 200,011 characters (kv_retrieval) to 1,116,721 (longbook_choice_eng).
- **PG-19.** Validation has 50 books, test has 100. Columns: `short_book_title`,
  `publication_date`, `url`, `text`. The train split is not on disk. The loader
  `pcm/tasks/pg19.py` yields one book at a time from the parquet file and raises an
  error if asked for the train split.

Reference code in `third_party/` (not tracked by git), used read-only for official
prompts and scoring:

| Repository | Commit | Commit date |
|---|---|---|
| booydar/babilong | `7a6efee29f5c` | 2026-06-01 |
| NVIDIA/RULER | `db43f92ff634` | 2026-11-03 |
| OpenBMB/InfiniteBench | `51d9b37b0f17` | 2024-09-26 |

## 5. Config check (`reports/config_check.json`)

| | Qwen3-1.7B | Qwen3-0.6B | Expected |
|---|---|---|---|
| num_hidden_layers | 28 | 28 | 28 |
| num_attention_heads | 16 | 16 | 16 |
| num_key_value_heads | 8 | 8 | 8 |
| head_dim | 128 | 128 | 128 |
| hidden_size | 2,048 | 1,024 | — |
| intermediate_size | 6,144 | 3,072 | — |
| vocab_size | 151,936 | 151,936 | — |
| rope_theta | 1,000,000 | 1,000,000 | — |
| max_position_embeddings | 40,960 | 40,960 | 40,960 (recorded, not asserted) |
| KV bytes per token, bf16 | 114,688 (112 KiB) | 114,688 (112 KiB) | 114,688 |
| Parameters | 1,720,574,976 | 596,049,920 | — |
| Non-embedding parameters | 1,409,410,048 | 440,467,456 | about 1.4B / 0.44B |
| Weights in bf16 | 3.20 GiB | 1.11 GiB | about 3.2 / 1.1 GiB |

All assertions passed. The memory tables of the specification (section 4.2) stand.

Three things the check surfaced that later steps need:

1. **RoPE API.** The shipped `config.json` files carry the older keys `rope_theta` and
   `rope_scaling: null`. The installed transformers (5.19.0) exposes them as
   `config.rope_parameters` with `rope_type: "default"`. YaRN will therefore be
   configured through `rope_parameters`, with 40,960 as the original length.
2. **Weight files hold a second copy of the embedding matrix.** The chat models store
   `lm_head.weight` separately although the weights are tied, so the files are larger
   than the model (2,031,739,904 stored values for 1.7B, 751,632,384 for 0.6B). The
   parameter counts above exclude the copy. The count from the weight files agrees
   exactly with a count rebuilt from the config, and with the loaded model (section 6).
3. **No sliding window.** `use_sliding_window` is false for both readers.

## 6. Smoke test (`reports/smoke.json`)

Reader `Qwen/Qwen3-1.7B`, bf16, on the GPU, SDPA attention, chat template with thinking
disabled, greedy decoding, `max_new_tokens=16`.

Prompt as sent to the model (19 tokens):

```text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Answer (8 new tokens): `The capital of France is Paris.`

| Measurement | Value |
|---|---|
| Parameters of the loaded model | 1,720,574,976 |
| Weights on the GPU | 3.20 GiB |
| Peak VRAM, reader only | 3.22 GiB |
| Embedder | Qwen/Qwen3-Embedding-0.6B, 1,024-dimensional vectors, bf16 |
| Cosine similarity of the two test sentences | 0.9119 |
| Peak VRAM, reader and embedder loaded together | 4.33 GiB allocated, 4.35 GiB reserved |
| Load time, reader | 11.58 s |

The two sentences were "The capital of France is Paris." and "Paris is the largest city
in France and its capital."

With thinking disabled the template inserts an empty think block after the assistant
header. This is the prompt form every later run uses.

The peak of 4.33 GiB is weights only (3.20 GiB + about 1.1 GiB) with a 19-token prompt.
It agrees with the weights and embedder columns of the VRAM table in section 4.2 of the
specification; it says nothing yet about the active-window KV cache.

## 7. Deviations

Recorded in `DEVIATIONS.md` as D-001 to D-008. In short:

- Nothing was downloaded: models, datasets and reference repositories were already on
  disk (D-001, D-002). The reference folder was renamed to `third_party/`.
- PyTorch is 2.14.1 with the CUDA 12.6 build, taken from the local package cache. 2.14.1
  is the newest version on the PyTorch index; a CUDA 13.0 build exists and was not used (D-003).
- Several packages are pinned to cached versions that are not the newest: transformers
  5.19.0 (newest 5.23.0) and huggingface_hub 1.33.0 (newest 2.4.0; datasets 5.0.0
  requires below 2.0) (D-004). Python is 3.11 (D-005).
- Files were added outside the planned folder tree: `pcm/hub.py`, `pcm/memory.py`,
  `pcm/tasks/pg19.py`, `configs/assets.yaml`, `scripts/fetch_assets.py`,
  `pyproject.toml` (D-006).
- The project stays on the Desktop, which is not OneDrive-synced on this machine (D-007).
- Parameters were counted from weight-file headers rather than by loading both models (D-008).

## 8. What is uncertain

- The page file's ability to grow past 32 GB was not verified.
- `HF_TOKEN` is not set. It was not needed, because nothing was downloaded. A gated or
  rate-limited download later would need it.
- The smoke test shows that the stack runs. It is not evidence about accuracy, long
  inputs, chunked prefill, YaRN or KV re-basing. Those are the subject of step A1.
- The server (Phase B) must reproduce these package versions from `requirements.lock.txt`
  with a Linux build of the same PyTorch version.

## 9. How to reproduce

```text
uv venv .venv --python 3.11
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv -r requirements.lock.txt
python scripts/hw_probe.py
python scripts/fetch_assets.py          # add --offline to forbid downloads
python scripts/check_config.py
python scripts/smoke.py
python -m pytest
```
