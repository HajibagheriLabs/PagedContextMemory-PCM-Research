# A1 — Core engine and its invariants

Date: 2026-11-14. Machine: local workstation (RTX 3090, 24 GB). Readers: Qwen/Qwen3-0.6B
and Qwen/Qwen3-1.7B.

This step builds the three pieces every later experiment stands on: loading a reader
(optionally with YaRN), prefilling a long input in blocks, and a container for KV caches
that can be sliced, concatenated and moved to a new position ("re-based"). Five tests
(T1 to T5) check the invariants. No task accuracy is measured here.

Every number below is read from a file under `results/A1_engine/`:
`tests_Qwen3-0.6B.json`, `tests_Qwen3-1.7B.json` (written by the tests) and
`throughput.jsonl` (written by `scripts/bench_prefill.py`). `scripts/a1_summary.py`
prints the tables. Every test record and every throughput row used here was measured on
commit `9274708`, with no uncommitted code changes. Test text is one PG-19 validation book, *The Atlantic Monthly Vol. 7
No. 42, April 1861*, the first book of that split with more than 100,000 tokens.

## 1. Outcome

| Test | What it checks | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| T1 | Chunked prefill equals one-shot prefill (6,000 tokens) | pass | pass |
| **T2** | **Re-based keys equal directly encoded keys, float32, max abs diff < 1e-3** | **FAIL** | **FAIL** |
| T2 diagnostics | Same comparison relative to scale; effect on logits; float64 | pass | pass |
| T3 | Stitched prefix + two pages + question: finite logits, same layout as text | pass | pass |
| T4 | YaRN factor 4 accepts 100,000 positions; losses recorded | pass | pass |
| T5 | 65,536 tokens prefilled in blocks under 12 GiB | pass | pass |

**T2 fails as specified, on both readers.** The specification says to stop and report in
that case, because every KV experiment depends on re-basing. Section 3 gives the
measurements, the cause as far as it was established, and a proposed replacement
criterion. The decision is the project owner's; until it is taken, re-basing is not
declared verified and no KV-mode experiment should build on it.

The remaining tests and the throughput measurements do not depend on that decision and
were completed.

Two things outside the tests need attention:

- The GPU dropped off the system once during this step (section 8). Nothing was lost:
  after it came back, both test suites and the benchmark were rerun on committed code and
  reproduced the earlier numbers. Runs are now built to continue after such a loss.
- T3 shows, on one prompt, how far a separately encoded page drifts from the same page
  encoded in sequence (section 4). It is a first look at the problem Gate G2 is about.

## 2. T1 — chunked prefill equals one-shot prefill

A 6,000-token input is encoded once in a single forward call and once in three blocks of
at most 2,048 tokens, each block reading the cache of the blocks before it.

| | Qwen3-0.6B | Qwen3-1.7B | Criterion |
|---|---|---|---|
| float32: max abs difference of last-position logits | 4.15e-05 | 3.81e-05 | < 1e-4 |
| float32: largest logit magnitude | 19.8 | 21.6 | |
| float32: max abs difference in cached keys | 0.00116 | 0.00434 | not asserted |
| float32: max abs difference in cached values | 0.00198 | 0.00468 | not asserted |
| bf16: top-1 token, one-shot / chunked | ` from` / ` from` | ` and` / ` and` | identical |
| bf16: max abs difference of last-position logits | 0.188 | 0.203 | not asserted |

T1 passes. One number here matters for T2: two computations that are mathematically
identical already differ by 0.001 to 0.005 in cached keys and values in float32. That is
the size of ordinary float32 noise for these tensors, and it is above T2's threshold.

## 3. T2 — re-basing: the specified test fails

**The test.** A 512-token page is encoded twice with no prefix: at positions 0–511 and
at positions 5000–5511. The keys of the first cache are then rotated by the rotary
transform for the constant position 5000. They should equal the keys of the second
cache; the values should be equal with no change. Float32, all 28 layers.

**What the library caches.** Read from the installed source (transformers 5.19.0,
`Qwen3Attention.forward`): keys are stored after the per-head key normalisation and
after the rotary transform; values carry no position. This is what the specification
assumed.

### 3.1 Result against the specified threshold

| | Qwen3-0.6B | Qwen3-1.7B | Criterion |
|---|---|---|---|
| Keys: max abs difference, worst layer | 0.0134 | 0.0158 | < 1e-3 |
| Values: max abs difference, worst layer | 0.00826 | 0.0270 | < 1e-3 |
| Layers (of 28) with key difference under 1e-3 | 1 | 5 | 28 |
| Layers (of 28) with value difference under 1e-3 | 12 | 9 | 28 |

**Fail.** The test is kept in the suite exactly as specified and keeps failing.

### 3.2 Why: what was measured

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Largest key magnitude per layer, range over layers | 22.3 – 478 | 22.2 – 358 |
| Largest value magnitude per layer, range over layers | 0.864 – 107 | 1.25 – 301 |
| Keys: max abs difference **without** re-basing | 65.4 | 76.6 |
| Keys: difference relative to the layer's largest key, worst layer | 2.69e-04 | 1.16e-04 |
| Values: same, worst layer | 1.83e-04 | 1.73e-04 |
| Next-token logits read through the re-based cache vs the directly encoded one, max abs difference | 1.43e-04 | 7.53e-05 |
| Same, with the cache left un-rebased | 2.43 | 5.66 |
| Top-1 token through re-based vs direct cache | same | same |
| Float64 model, library's float32 angles: keys max abs difference | 0.0141 | 0.0157 |
| Float64 model, float64 angles: keys max abs difference | 7.59e-06 | 2.78e-05 |
| Float64 model, float64 angles: keys, relative | 1.81e-07 | 7.25e-07 |
| Float64 model, float64 angles: values, relative | 2.00e-07 | 4.22e-07 |

The logits row is from 64 further tokens of the book read after the page, at positions
5512 onward, once through each cache.

What these rows establish:

1. **The threshold is absolute and the tensors are large.** Cached keys reach
   magnitudes of several hundred. A difference of 0.013 on a tensor whose entries reach
   478 is 2.7e-04 of its scale. It is also within a factor of 4 to 12 of the float32
   noise T1 measured between two identical computations.
2. **Re-basing does what it should.** Without it the keys differ by 65 to 77 and the
   logits by 2.4 to 5.7. With it the logits differ by about 1e-04 and the top-1 token is
   the same.
3. **The remaining difference comes from how the library computes rotary angles, not
   from re-basing.** The library multiplies position by frequency in float32 whatever
   the model's precision. Running the whole model in float64 but leaving that step alone
   reproduces the float32 result (0.0141 and 0.0157, against 0.0134 and 0.0158).
   Computing the angles in float64 as well makes the difference fall by three orders of
   magnitude, to 1.8e-07 and 7.3e-07 of scale. In that run the rotary module was patched
   inside the test only; the engine uses the library unchanged.

The reading that follows: at positions near 5000 a float32 angle carries rounding error,
so the model's own output is not exactly the same at two different absolute positions.
Re-basing is exact algebra applied to slightly rounded inputs. This reading rests on the
three rows above; it was not otherwise proven.

One observation is not explained. In the float64 run with float64 angles the keys agree
to about 1e-13 in the lower layers and then to between 1e-06 and 3e-05 from some layer upward
(layer 19 on Qwen3-0.6B, layer 5 on Qwen3-1.7B). The agreement is still better than
1e-06 of scale, but why it steps at those layers was not investigated.

### 3.3 Proposed fix, for the owner's decision

Replace T2's criterion by a scale-aware one and keep the float64 check as the proof of
exactness:

- per layer, max abs difference divided by the largest magnitude in the same tensor
  < 1e-3 (observed worst case 2.7e-04);
- the next-token logits through the re-based cache within 1e-2 of those through the
  directly encoded cache, with the same top-1 token (observed 1.4e-04);
- float64 with float64 angles: relative difference < 1e-6 (observed 7.3e-07).

These three checks already run as `test_t2_relative_and_functional` and
`test_t2_exact_in_float64` and pass on both readers. **Their thresholds were chosen
after seeing the first results**, so passing them is a description of what was measured,
not an independent confirmation. The margin on the float64 check is small on Qwen3-1.7B
(7.3e-07 against 1e-06).

No change to the engine is proposed. Forcing float64 angles inside the model would alter
the reader for every method and is not justified by a deviation this small.

## 4. T3 — stitching sanity

A shared prefix (system prompt, instruction, "Context:") is encoded once. Two 512-token
pages from the book, taken 2,048 tokens apart, are each encoded as a continuation of the
prefix. Their caches are concatenated after the prefix, the second one re-based by 512
positions, and the question is then read. The same token sequence is also encoded as one
ordinary text prompt.

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Tokens: prefix / page / page / question part | 24 / 512 / 512 / 22 | 24 / 512 / 512 / 22 |
| Total tokens, KV mode / text mode | 1,070 / 1,070 | 1,070 / 1,070 |
| Positions of the question part, KV mode / text mode | 1048–1069 / 1048–1069 | 1048–1069 / 1048–1069 |
| Logits finite | yes | yes |
| Layer-0 keys, stitched vs text, relative difference | 2.00e-06 | 1.16e-06 |
| Layer-0 values, stitched vs text, relative difference | 1.15e-06 | 4.31e-07 |
| First page, keys at all layers, relative difference | 4.01e-06 | 3.15e-05 |
| Second page, keys at layer 0, relative difference | 1.37e-06 | 1.28e-06 |
| Second page, keys at layers 1–27, Frobenius relative difference, range | 0.019 – 0.375 | 0.031 – 0.393 |
| Last-position logits, KV vs text, max abs difference | 1.90 | 2.32 |
| Top-1 token, KV / text | `The` / `The` | `The` / `The` |

T3 passes: token count and positions match, and logits are finite. The layout check is
numerical: at layer 0 a key or value depends only on the token and its position, so
agreement there at every position shows each stitched token sits where the text prompt
puts it. The first page sits directly after the prefix in both modes and agrees at every
layer. The thresholds (1e-3 relative) are this report's own; the specification names the
property but not how to check it.

The second page is the informative part. It was encoded without seeing the first page,
and above layer 0 its keys differ from sequential encoding by 2% to 39% in Frobenius
norm depending on the layer. The final logits move by about 2 while the top-1 token
stays the same. This is one prompt and not an accuracy measurement. It shows that the
known cost of independent encoding is present and not small, which is the question
experiment E6 and Gate G2 exist to answer.

## 5. T4 — YaRN

`load_reader(model, yarn_factor=4)` configures YaRN through `config.rope_parameters`,
the API of the installed library, with `rope_theta` (1,000,000) and the original length
(40,960) read from the shipped config. The loaded model reports rope type `yarn`, factor
4, maximum positions 163,840 and an attention scaling of 1.1386.

Next-token loss (nats per token, bf16) on the same book. All four conditions end on the
same token: the 100,000-token window is fixed and a condition of n tokens uses its last
n. "Tail" is the mean over the last 2,048 tokens, which are the same tokens in every
condition; "all" is the mean over every predicted token of that condition.

| Condition | Tokens | 0.6B tail | 0.6B all | 1.7B tail | 1.7B all | Peak VRAM 0.6B / 1.7B (GiB) |
|---|---|---|---|---|---|---|
| Native RoPE | 16,384 | 3.785 | 3.731 | 3.445 | 3.423 | 5.19 / 7.29 |
| Native RoPE, past its range | 49,152 | 3.871 | 3.663 | 3.533 | 3.380 | 8.69 / 10.79 |
| YaRN, factor 4 | 49,152 | 4.526 | 4.314 | 3.815 | 3.686 | 8.69 / 10.79 |
| YaRN, factor 4 | 100,000 | 4.458 | 4.270 | 3.798 | 3.672 | 14.16 / 16.26 |

T4 passes: with factor 4 both readers accept 100,000 positions and every loss is finite.

What the losses say, with the limits stated first: one book, one window, no confidence
interval, and next-token loss on prose is not the task metric.

- On the same 2,048 tokens, training-free YaRN is worse than native RoPE at 49,152
  tokens: +0.66 nats on Qwen3-0.6B and +0.28 on Qwen3-1.7B.
- Native RoPE at 49,152 tokens, about 8,000 positions past the 40,960 it was configured
  for, is only +0.09 worse on the tail than at 16,384 tokens on both readers.
- YaRN at 100,000 tokens is no worse than YaRN at 49,152.

This bears on the dense baseline M1, which uses training-free YaRN. If the same pattern
holds on tasks, M1 may be weak at every length, and the specification's risk table
already says what to do then (report it and add the fine-tuned baseline M4, so H1 is not
won by default). Native RoPE without YaRN at 100,000 tokens was not measured here.

## 6. T5 — memory guard, and greedy generation

65,536 tokens of the book are prefilled in blocks of 2,048, bf16.

| | Qwen3-0.6B | Qwen3-1.7B | Criterion |
|---|---|---|---|
| Allocated before (weights) | 1.12 GiB | 3.21 GiB | |
| Cache produced | 7.00 GiB | 7.00 GiB | |
| Peak VRAM, allocated | 9.02 GiB | 11.13 GiB | < 12 GiB |
| Peak VRAM, reserved by the allocator | 9.74 GiB | 11.83 GiB | not asserted |

T5 passes on both readers. On Qwen3-1.7B the margin is 0.87 GiB.

Two engineering findings came out of this test:

- **Reserved memory ran far above allocated memory.** In the first version, reserved
  memory reached 20.4 GiB at 65,536 tokens on Qwen3-0.6B while only 9.0 GiB was
  allocated. Each block needs slightly larger temporaries than the block before, so the
  allocator cannot reuse what it just freed. `pcm/prefill.py` now returns unused blocks
  to the driver after a block whenever reserved exceeds allocated by more than 1 GiB.
  Reserved then stayed within 0.8 GiB of allocated, at the same speed. Without this, a
  dense 128K run (14 GiB of cache) would be at risk on a 24 GB card. The 20.4 GiB figure
  was observed during development and is not in a results file.
- **The cache is written into one preallocated buffer.** The library grows its cache by
  building a longer tensor on every call, which copies the whole cache once per block.
  The adapter in `pcm/store/kv.py` supplies a cache layer that allocates the final size
  once and writes each block in place.

Greedy generation (`generate_greedy`) was checked against the library's own `generate`
on the question "What is the capital of France?", with the prompt split between a
prefilled cache and the remainder. The token sequences are identical on both readers.
Answers: `The capital of France is **Paris**.` (0.6B) and `The capital of France is
Paris.` (1.7B).

## 7. Prefill throughput

Chunked prefill of the first n tokens of the book, blocks of 2,048, bf16, three timed
runs per cell after one warm-up. Median of three; the spread is the minimum and maximum.

| Model | Tokens | Tokens/s (median) | Spread | Seconds | Peak VRAM allocated | Reserved |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 8,192 | 14,337 | 13,849 – 14,352 | 0.57 | 2.13 GiB | 2.29 GiB |
| Qwen3-0.6B | 32,768 | 5,945 | 5,916 – 5,956 | 5.51 | 5.08 GiB | 5.80 GiB |
| Qwen3-0.6B | 65,536 | 3,309 | 3,309 – 3,316 | 19.81 | 9.02 GiB | 9.73 GiB |
| Qwen3-1.7B | 8,192 | 9,776 | 9,719 – 9,874 | 0.84 | 4.24 GiB | 4.61 GiB |
| Qwen3-1.7B | 32,768 | 4,947 | 4,941 – 4,953 | 6.62 | 7.19 GiB | 7.90 GiB |
| Qwen3-1.7B | 65,536 | 2,979 | 2,977 – 2,981 | 22.00 | 11.13 GiB | 11.83 GiB |

Reading:

- Throughput falls with length because each block attends to everything before it.
  Doubling from 32,768 to 65,536 tokens takes 3.6 times as long on Qwen3-0.6B and 3.3
  times on Qwen3-1.7B.
- The file also holds 18 earlier rows measured before the code was committed. They are
  not used in the table; their medians are within 4% of the ones shown.
- Qwen3-1.7B takes 1.47 times as long as Qwen3-0.6B at 8,192 tokens, 1.20 times at
  32,768 and 1.11 times at 65,536. The specification's arithmetic expected about 1.5
  for a window-sized input and about 1.1 for a long dense pass.
- A full window costs about 6.6 seconds to prefill on Qwen3-1.7B. That is roughly the
  per-question cost of text expansion with a full page budget, before generation.
- Peak memory is weights plus cache plus about 0.9 GiB of working memory.

## 8. GPU loss, and how runs now survive it

**What happened.** At 10:38:43 Windows logged that the NVIDIA display driver stopped
responding. From then until the device was restored, the RTX 3090 was invisible to
`nvidia-smi` and to PyTorch. All tests and the benchmark had finished about a minute
earlier and nothing from this project was running on the GPU at that moment. The cause
is unknown. The owner reports the same kind of loss in other work on this machine.

**What it cost here.** The first set of measurements had been taken just before the code
was committed. The rerun meant to stamp them with the committed code found no GPU,
skipped all nine GPU tests without failing, and overwrote the saved console logs of the
real runs. No measurement was lost, but the skipped run looked like a success.

**After the GPU came back** both suites and the benchmark were rerun on commit
`9274708`. Every test number in this report is identical to the first run at the
precision shown. Throughput is 2% to 3% higher than in the first run. This report uses the rerun only.

**What changed so that a loss costs minutes and nothing has to be redone:**

| Risk | Now |
|---|---|
| A crash in the middle of writing a results file | Result rows are appended one per unit of work and forced to disk before the run moves on (`pcm/runlog.py`). A line cut short by a crash is ignored on reading and never extended. Other files are written under a temporary name and renamed into place |
| Losing finished work | A rerun of the same command reads the ids already present and skips them. Test records and benchmark rows carry the commit they were measured on; only work done on the current, clean commit counts as done |
| A lost GPU recorded as a wrong answer or a failed test | An exception is classified as lost GPU, out of memory, or ordinary error (`pcm/health.py`). A lost GPU stops the run at once, writes `interrupted_*.json` naming the last finished unit, and exits with status 75. Results are left untouched |
| A missing GPU passing silently | The test session stops with status 75 if the GPU is absent. Skipping GPU tests now requires asking for it (`--no-gpu`) |
| Starting a run that cannot finish | A preflight check of the GPU, free VRAM, RAM and disk runs first |
| Not knowing what the card was doing before a crash | Temperature, power, load, memory and fan speed are sampled every 10 seconds to `logs/<experiment>/gpu_telemetry.csv` and flushed as written |

Resuming: `python -m pytest --reader <reader> --resume` continues a test session;
`python scripts/bench_prefill.py` continues the benchmark. The rule for later steps is
that the work between two durable writes takes minutes at most: one example in an
evaluation, one document in ingest, a fixed number of steps in training.

What this does not cover: a loss in the middle of one long unit repeats that unit (the
longest here is T4, a few minutes). The resume path was exercised by unit tests on
the log and by the benchmark skipping finished rows; it has not been exercised by a real
GPU loss in mid-run.

During the rerun the card peaked at 64 °C and 273 W. Those two figures come from the
telemetry log, which is not tracked in the repository.

Recorded as D-014 to D-016 in `DEVIATIONS.md`.

## 9. What was built

| File | Contents |
|---|---|
| `pcm/models.py` | `load_reader` (local cache, optional YaRN, detects the RoPE API), `build_chat` (chat template, thinking disabled, prompt split into prefix / context / suffix), helpers |
| `pcm/prefill.py` | `prefill_in_blocks`, `generate_greedy`, `nll_in_blocks` |
| `pcm/store/kv.py` | `KVCache` (per-layer access, slice, concatenate, move, convert to and from the library cache), `rebase_keys`, `encode_page`, `stitch`; the only module that touches library cache objects |
| `pcm/env.py` | Environment record for result files |
| `pcm/runlog.py` | Durable append-only result log; atomic file writes |
| `pcm/health.py` | GPU preflight, lost-GPU classification, telemetry sampler |
| `tests/` | 46 tests: 37 on CPU (adapter, re-basing algebra, result log, failure classification, memory arithmetic, assets) and 9 on GPU (T1–T5 and diagnostics). Final run on each reader: 45 passed, 1 failed (T2 as specified) |
| `configs/a1_engine.yaml` | Every parameter and threshold of the tests and the benchmark |
| `scripts/bench_prefill.py`, `scripts/a1_summary.py` | Throughput benchmark; tables of this report |

Design points later steps rely on:

- Prompts are assembled in token space: prefix ids + context ids + suffix ids. Text mode
  and KV mode therefore see the same tokens at the same positions.
- The LM head is applied only at positions whose logits are needed.
- Re-basing takes the frequencies from the loaded model, never from a formula.

Departures from the listed function signatures and the added functions are in
`DEVIATIONS.md` (D-010 to D-013).

## 10. What is uncertain

- **T2**, pending the owner's decision (section 3.3).
- The step in float64 agreement at upper layers (section 3.2) is unexplained.
- The cause of the GPU loss is unknown, so it may recur at any time (section 8).
- T4 uses one book and one window. Its direction may not carry over to tasks.
- T3's deviation figures are from one prompt with two pages.
- Whether the model accepts a custom 4D attention mask unchanged was read in the library
  source but not tested. The APE-style scaling factor in A6 needs it.
- `prefill_in_blocks` copies an existing cache once into its buffer. Reading a question
  after a full stitched window therefore holds two copies of that window for a moment
  (about 7 GiB at 32K). This fits the card but should be removed before KV mode is timed.
- Throughput was measured on prose with a quiet machine. No other length was measured.
- The saved console logs of the test runs are in `logs/`, which is not tracked.
