# A4 tables (written by scripts/a4_summary.py from the results files)

Code commits of the rows: {"E0_a4budget": {"ff31198f94": 300}, "E0_floor": {"2502764511": 20, "db85c9008c": 15230}, "E1_truncation": {"db85c9008c": 580, "207e80e6e4": 11020, "6db01c5ea2": 600}, "E2_dense": {"ff31198f94": 1000}, "E3_retrieval": {"ff31198f94": 22400}, "E3o_oracle": {"ff31198f94": 13600}}

Rows left out, run with settings since replaced (D-040): {"E2_dense|m1-yarn-1.7B": 20, "E3_retrieval|m2-hybrid-1.7B": 25, "E3_retrieval|m2-hybrid-rerank-1.7B": 25, "E3_retrieval|m2-bm25-1.7B": 25, "E3_retrieval|m2-embed-1.7B": 25, "E3o_oracle|m3-oracle-tight-1.7B": 25, "E3o_oracle|m3-oracle-1.7B": 15}

References (native 8K, mean of the family's tasks): S1 100.0 [100.0, 100.0] (A3: 100.0 [100.0, 100.0]); S2 82.5 [77.0, 87.5] (A3: 82.5 [77.0, 87.5]); BABILong 57.2 [53.2, 61.2] (A3: 57.2 [53.2, 61.2]); BABILong-ext 53.4 [50.0, 56.8] (A3: 53.4 [50.0, 56.8])

## S1: accuracy (%, 95% CI), n per task in brackets. Reference (native 8K): 100.0 [100.0, 100.0]; 90% of it: 90.00

| Method | 8k | 16k | 32k | 64k | 128k | 256k | ECL90 | stops at |
|---|---|---|---|---|---|---|---|---|
| M0 last 32K | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 50.0 [40.0, 60.0] (n=100) | 26.0 [18.0, 35.0] (n=100) | 12.0 [6.0, 19.0] (n=100) | 32k | 64k: below threshold |
| M0 first + last 16K | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 50.0 [40.0, 59.0] (n=100) | 23.0 [15.0, 31.0] (n=100) | 14.0 [8.0, 21.0] (n=100) | 32k | 64k: below threshold |
| M1 dense YaRN | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (n=40) | 100.0 [100.0, 100.0] (n=40) | – | 128k | 256k: not measured |
| M2 BM25 | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (n=100) | 99.0 [97.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |
| M2 embeddings | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |
| M2 hybrid | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 99.0 [97.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |
| M2 hybrid + reranker | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 99.0 [97.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |
| M3 oracle | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (native) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 99.0 [97.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |
| M3 oracle, tight (supplementary) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 100.0 [100.0, 100.0] (n=100) | 256k | end of grid |

## S2: accuracy (%, 95% CI), n per task in brackets. Reference (native 8K): 82.5 [77.0, 87.5]; 90% of it: 74.25

| Method | 8k | 16k | 32k | 64k | 128k | 256k | ECL90 | stops at |
|---|---|---|---|---|---|---|---|---|
| M0 last 32K | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 64.5 [58.0, 71.0] (native) | 26.5 [21.0, 32.0] (n=100) | 11.5 [7.5, 15.5] (n=100) | 6.0 [3.0, 9.5] (n=100) | 16k | 32k: below threshold |
| M0 first + last 16K | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 64.5 [58.0, 71.0] (native) | 20.5 [15.5, 25.5] (n=100) | 10.0 [6.0, 14.0] (n=100) | 7.5 [4.0, 11.0] (n=100) | 16k | 32k: below threshold |
| M1 dense YaRN | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 64.5 [58.0, 71.0] (native) | 62.5 [52.5, 72.5] (n=40) | 31.2 [21.2, 41.2] (n=40) | – | 16k | 32k: below threshold |
| M2 BM25 | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 68.5 [62.0, 74.5] (n=100) | 74.0 [68.0, 80.0] (n=100) | 74.0 [68.0, 80.0] (n=100) | 79.0 [73.0, 84.5] (n=100) | 16k | 32k: below threshold |
| M2 embeddings | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 68.0 [61.5, 74.5] (n=100) | 75.0 [69.0, 81.0] (n=100) | 74.0 [68.0, 80.0] (n=100) | 77.0 [71.0, 82.5] (n=100) | 16k | 32k: below threshold |
| M2 hybrid | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 63.5 [57.0, 70.0] (n=100) | 76.0 [70.0, 81.5] (n=100) | 70.0 [63.5, 76.0] (n=100) | 77.5 [71.5, 83.0] (n=100) | 16k | 32k: below threshold |
| M2 hybrid + reranker | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 63.5 [57.0, 70.0] (n=100) | 76.0 [70.0, 81.5] (n=100) | 70.0 [63.5, 76.0] (n=100) | 77.5 [71.5, 83.0] (n=100) | 16k | 32k: below threshold |
| M3 oracle | 82.5 [77.0, 87.5] (native) | 78.5 [72.5, 84.0] (native) | 69.5 [63.5, 75.5] (n=100) | 73.5 [68.0, 79.0] (n=100) | 79.0 [73.5, 84.0] (n=100) | 80.0 [75.0, 85.0] (n=100) | 16k | 32k: below threshold |
| M3 oracle, tight (supplementary) | 95.0 [92.0, 97.5] (n=100) | 94.5 [91.5, 97.5] (n=100) | 95.0 [92.0, 97.5] (n=100) | 98.0 [96.0, 99.5] (n=100) | 97.0 [94.5, 99.0] (n=100) | 97.5 [95.0, 99.5] (n=100) | 256k | end of grid |

## BABILong: accuracy (%, 95% CI), n per task in brackets. Reference (native 8K): 57.2 [53.2, 61.2]; 90% of it: 51.48

| Method | 8k | 16k | 32k | 64k | 128k | 256k | ECL90 | stops at |
|---|---|---|---|---|---|---|---|---|
| M0 last 32K | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 43.9 [39.8, 47.9] (native) | 40.0 [35.8, 44.2] (n=100) | 31.0 [27.0, 35.0] (n=100) | 26.2 [22.4, 30.0] (n=100) | 8k | 16k: below threshold |
| M0 first + last 16K | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 43.9 [39.8, 47.9] (native) | 39.2 [35.0, 43.4] (n=100) | 27.6 [23.8, 31.4] (n=100) | 24.8 [21.0, 28.6] (n=100) | 8k | 16k: below threshold |
| M1 dense YaRN | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 43.9 [39.8, 47.9] (native) | 44.0 [37.5, 50.5] (n=40) | 26.0 [20.0, 32.0] (n=40) | – | 8k | 16k: below threshold |
| M2 BM25 | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 45.0 [41.0, 49.0] (n=100) | 38.2 [34.2, 42.2] (n=100) | 37.0 [33.2, 40.8] (n=100) | 34.0 [30.4, 37.8] (n=100) | 8k | 16k: below threshold |
| M2 embeddings | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 42.4 [38.4, 46.6] (n=100) | 38.6 [34.6, 42.6] (n=100) | 32.8 [29.0, 36.6] (n=100) | 29.4 [25.6, 33.2] (n=100) | 8k | 16k: below threshold |
| M2 hybrid | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 42.0 [38.0, 46.0] (n=100) | 39.6 [35.6, 43.6] (n=100) | 33.6 [30.0, 37.4] (n=100) | 33.0 [29.2, 36.8] (n=100) | 8k | 16k: below threshold |
| M2 hybrid + reranker | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 42.0 [38.0, 46.0] (n=100) | 39.6 [35.6, 43.6] (n=100) | 33.6 [30.0, 37.4] (n=100) | 33.0 [29.2, 36.8] (n=100) | 8k | 16k: below threshold |
| M3 oracle | 57.2 [53.2, 61.2] (native) | 42.6 [38.6, 46.4] (native) | 42.0 [38.0, 46.0] (n=100) | 45.4 [41.2, 49.6] (n=100) | 45.0 [40.8, 49.2] (n=100) | 44.8 [40.6, 48.8] (n=100) | 8k | 16k: below threshold |
| M3 oracle, tight (supplementary) | 58.4 [54.4, 62.6] (n=100) | 57.2 [53.2, 61.2] (n=100) | 56.0 [52.0, 60.0] (n=100) | 57.6 [53.4, 61.8] (n=100) | 56.2 [52.0, 60.4] (n=100) | 55.8 [51.6, 60.0] (n=100) | 256k | end of grid |

## BABILong-ext: accuracy (%, 95% CI), n per task in brackets. Reference (native 8K): 53.4 [50.0, 56.8]; 90% of it: 48.06

| Method | 8k | 16k | 32k | 64k | 128k | 256k | ECL90 | stops at |
|---|---|---|---|---|---|---|---|---|
| M0 last 32K | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 44.6 [40.9, 48.2] (native) | 41.4 [37.6, 45.0] (n=100) | 40.0 [36.4, 43.8] (n=100) | 37.8 [34.2, 41.6] (n=100) | none | 8k: below floor |
| M0 first + last 16K | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 44.6 [40.9, 48.2] (native) | 38.6 [35.0, 42.2] (n=100) | 37.4 [33.6, 41.4] (n=100) | 37.2 [33.4, 41.0] (n=100) | none | 8k: below floor |
| M1 dense YaRN | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 44.6 [40.9, 48.2] (native) | 38.7 [32.0, 45.3] (n=30) | 32.0 [26.0, 38.7] (n=30) | – | none | 8k: below floor |
| M2 BM25 | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 41.4 [37.8, 45.2] (n=100) | 41.2 [37.6, 44.8] (n=100) | 41.0 [37.4, 44.6] (n=100) | 43.2 [39.6, 47.0] (n=100) | none | 8k: below floor |
| M2 embeddings | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 42.2 [38.6, 45.8] (n=100) | 40.4 [36.6, 44.0] (n=100) | 41.6 [37.8, 45.2] (n=100) | 42.0 [38.4, 45.8] (n=100) | none | 8k: below floor |
| M2 hybrid | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 40.8 [37.2, 44.4] (n=100) | 40.4 [36.8, 44.0] (n=100) | 40.8 [37.0, 44.6] (n=100) | 42.8 [39.2, 46.4] (n=100) | none | 8k: below floor |
| M2 hybrid + reranker | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 40.8 [37.2, 44.4] (n=100) | 40.4 [36.8, 44.0] (n=100) | 40.8 [37.0, 44.6] (n=100) | 42.8 [39.2, 46.4] (n=100) | none | 8k: below floor |
| M3 oracle | 53.4 [50.0, 56.8] (native) | 45.2 [41.6, 48.8] (native) | 40.8 [37.0, 44.4] (n=100) | 41.0 [37.2, 44.8] (n=100) | 39.8 [36.2, 43.4] (n=100) | 40.2 [36.6, 43.8] (n=100) | none | 8k: below floor |
| M3 oracle, tight (supplementary) | 55.8 [52.2, 59.2] (n=100) | 56.2 [52.6, 59.8] (n=100) | 55.2 [51.8, 58.6] (n=100) | 55.6 [52.0, 59.0] (n=100) | 56.8 [53.4, 60.4] (n=100) | 55.0 [51.6, 58.4] (n=100) | none | 8k: below floor |

## ECL90 by family

| Method | S1 | S2 | BABILong | BABILong-ext |
|---|---|---|---|---|
| M0 last 32K | 32k | 16k | 8k | none |
| M0 first + last 16K | 32k | 16k | 8k | none |
| M1 dense YaRN | 128k | 16k | 8k | none |
| M2 BM25 | 256k | 16k | 8k | none |
| M2 embeddings | 256k | 16k | 8k | none |
| M2 hybrid | 256k | 16k | 8k | none |
| M2 hybrid + reranker | 256k | 16k | 8k | none |
| M3 oracle | 256k | 16k | 8k | none |
| M3 oracle, tight (supplementary) | 256k | 256k | 256k | none |

## Ratio condition alone (accuracy >= 90% of the reference at every length up to L); not an ECL90 (D-032)

| Method | S1 | S2 | BABILong | BABILong-ext |
|---|---|---|---|---|
| M0 last 32K | 32k | 16k | 8k | 8k |
| M0 first + last 16K | 32k | 16k | 8k | 8k |
| M1 dense YaRN | 128k | 16k | 8k | 8k |
| M2 BM25 | 256k | 16k | 8k | 8k |
| M2 embeddings | 256k | 16k | 8k | 8k |
| M2 hybrid | 256k | 16k | 8k | 8k |
| M2 hybrid + reranker | 256k | 16k | 8k | 8k |
| M3 oracle | 256k | 16k | 8k | 8k |
| M3 oracle, tight (supplementary) | 256k | 256k | 256k | 256k |

## Paired differences (points, 95% CI; first minus second)

| Pair | Family | Length | n | Difference | Excludes 0 |
|---|---|---|---|---|---|
| M3 oracle minus M2 hybrid | S1 | 32k | 100 | +1.0 [+0.0, +3.0] | no |
| M3 oracle minus M2 hybrid | S1 | 64k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle minus M2 hybrid | S1 | 128k | 100 | -1.0 [-3.0, +0.0] | no |
| M3 oracle minus M2 hybrid | S1 | 256k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle minus M2 hybrid | S2 | 32k | 200 | +6.0 [-0.5, +12.5] | no |
| M3 oracle minus M2 hybrid | S2 | 64k | 200 | -2.5 [-9.5, +4.0] | no |
| M3 oracle minus M2 hybrid | S2 | 128k | 200 | +9.0 [+1.0, +17.0] | yes |
| M3 oracle minus M2 hybrid | S2 | 256k | 200 | +2.5 [-5.0, +10.0] | no |
| M3 oracle minus M2 hybrid | BABILong | 32k | 500 | +0.0 [-2.8, +3.0] | no |
| M3 oracle minus M2 hybrid | BABILong | 64k | 500 | +5.8 [+1.6, +10.0] | yes |
| M3 oracle minus M2 hybrid | BABILong | 128k | 500 | +11.4 [+6.6, +16.0] | yes |
| M3 oracle minus M2 hybrid | BABILong | 256k | 500 | +11.8 [+7.2, +16.4] | yes |
| M3 oracle minus M2 hybrid | BABILong-ext | 32k | 500 | +0.0 [-2.2, +2.2] | no |
| M3 oracle minus M2 hybrid | BABILong-ext | 64k | 500 | +0.6 [-2.2, +3.6] | no |
| M3 oracle minus M2 hybrid | BABILong-ext | 128k | 500 | -1.0 [-4.2, +2.2] | no |
| M3 oracle minus M2 hybrid | BABILong-ext | 256k | 500 | -2.6 [-6.2, +1.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 8k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 16k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 32k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 64k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 128k | 100 | +1.0 [+0.0, +3.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S1 | 256k | 100 | +0.0 [+0.0, +0.0] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 8k | 200 | +12.5 [+7.0, +18.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 16k | 200 | +16.0 [+10.0, +22.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 32k | 200 | +25.5 [+18.5, +32.5] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 64k | 200 | +24.5 [+18.5, +31.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 128k | 200 | +18.0 [+12.5, +24.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | S2 | 256k | 200 | +17.5 [+12.0, +23.5] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 8k | 500 | +1.2 [-2.0, +4.4] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 16k | 500 | +14.6 [+10.4, +18.6] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 32k | 500 | +14.0 [+9.6, +18.4] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 64k | 500 | +12.2 [+7.6, +16.8] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 128k | 500 | +11.2 [+6.6, +16.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong | 256k | 500 | +11.0 [+6.2, +15.8] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 8k | 500 | +2.4 [-0.4, +5.4] | no |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 16k | 500 | +11.0 [+7.0, +15.0] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 32k | 500 | +14.4 [+9.6, +19.2] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 64k | 500 | +14.6 [+10.0, +19.2] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 128k | 500 | +17.0 [+12.4, +21.6] | yes |
| M3 oracle, tight (supplementary) minus M3 oracle | BABILong-ext | 256k | 500 | +14.8 [+10.4, +19.2] | yes |
| M2 hybrid minus M0 last 32K | S1 | 32k | 100 | -1.0 [-3.0, +0.0] | no |
| M2 hybrid minus M0 last 32K | S1 | 64k | 100 | +50.0 [+40.0, +60.0] | yes |
| M2 hybrid minus M0 last 32K | S1 | 128k | 100 | +74.0 [+65.0, +82.0] | yes |
| M2 hybrid minus M0 last 32K | S1 | 256k | 100 | +88.0 [+81.0, +94.0] | yes |
| M2 hybrid minus M0 last 32K | S2 | 32k | 200 | -1.0 [-7.0, +5.0] | no |
| M2 hybrid minus M0 last 32K | S2 | 64k | 200 | +49.5 [+41.5, +57.5] | yes |
| M2 hybrid minus M0 last 32K | S2 | 128k | 200 | +58.5 [+50.5, +66.0] | yes |
| M2 hybrid minus M0 last 32K | S2 | 256k | 200 | +71.5 [+65.0, +77.5] | yes |
| M2 hybrid minus M0 last 32K | BABILong | 32k | 497 | -1.8 [-5.2, +1.4] | no |
| M2 hybrid minus M0 last 32K | BABILong | 64k | 500 | -0.4 [-5.0, +4.2] | no |
| M2 hybrid minus M0 last 32K | BABILong | 128k | 500 | +2.6 [-2.0, +7.4] | no |
| M2 hybrid minus M0 last 32K | BABILong | 256k | 500 | +6.8 [+1.8, +11.8] | yes |
| M2 hybrid minus M0 last 32K | BABILong-ext | 32k | 495 | -3.8 [-6.9, -0.8] | yes |
| M2 hybrid minus M0 last 32K | BABILong-ext | 64k | 500 | -1.0 [-4.6, +2.6] | no |
| M2 hybrid minus M0 last 32K | BABILong-ext | 128k | 500 | +0.8 [-2.8, +4.2] | no |
| M2 hybrid minus M0 last 32K | BABILong-ext | 256k | 500 | +5.0 [+1.4, +8.6] | yes |
| M2 hybrid minus M1 dense YaRN | S1 | 32k | 100 | -1.0 [-3.0, +0.0] | no |
| M2 hybrid minus M1 dense YaRN | S1 | 64k | 40 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M1 dense YaRN | S1 | 128k | 40 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M1 dense YaRN | S2 | 32k | 200 | -1.0 [-7.0, +5.0] | no |
| M2 hybrid minus M1 dense YaRN | S2 | 64k | 80 | +8.8 [-5.0, +22.5] | no |
| M2 hybrid minus M1 dense YaRN | S2 | 128k | 80 | +40.0 [+26.2, +52.5] | yes |
| M2 hybrid minus M1 dense YaRN | BABILong | 32k | 497 | -1.8 [-5.2, +1.4] | no |
| M2 hybrid minus M1 dense YaRN | BABILong | 64k | 200 | -0.5 [-7.5, +6.5] | no |
| M2 hybrid minus M1 dense YaRN | BABILong | 128k | 200 | +9.5 [+2.5, +16.5] | yes |
| M2 hybrid minus M1 dense YaRN | BABILong-ext | 32k | 495 | -3.8 [-6.9, -0.8] | yes |
| M2 hybrid minus M1 dense YaRN | BABILong-ext | 64k | 150 | +2.7 [-4.7, +10.0] | no |
| M2 hybrid minus M1 dense YaRN | BABILong-ext | 128k | 150 | +7.3 [+0.0, +14.7] | no |
| M1 dense YaRN minus M0 last 32K | S1 | 64k | 40 | +50.0 [+35.0, +65.0] | yes |
| M1 dense YaRN minus M0 last 32K | S1 | 128k | 40 | +77.5 [+65.0, +90.0] | yes |
| M1 dense YaRN minus M0 last 32K | S2 | 64k | 80 | +35.0 [+20.0, +50.0] | yes |
| M1 dense YaRN minus M0 last 32K | S2 | 128k | 80 | +21.2 [+8.8, +33.8] | yes |
| M1 dense YaRN minus M0 last 32K | BABILong | 64k | 200 | +3.0 [-4.5, +10.5] | no |
| M1 dense YaRN minus M0 last 32K | BABILong | 128k | 200 | -3.0 [-10.5, +4.5] | no |
| M1 dense YaRN minus M0 last 32K | BABILong-ext | 64k | 150 | +0.0 [-8.0, +8.0] | no |
| M1 dense YaRN minus M0 last 32K | BABILong-ext | 128k | 150 | -4.7 [-13.3, +4.0] | no |
| M2 hybrid minus M2 BM25 | S1 | 32k | 100 | -1.0 [-3.0, +0.0] | no |
| M2 hybrid minus M2 BM25 | S1 | 64k | 100 | +1.0 [+0.0, +3.0] | no |
| M2 hybrid minus M2 BM25 | S1 | 128k | 100 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M2 BM25 | S1 | 256k | 100 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M2 BM25 | S2 | 32k | 200 | -5.0 [-11.0, +1.0] | no |
| M2 hybrid minus M2 BM25 | S2 | 64k | 200 | +2.0 [-4.5, +8.5] | no |
| M2 hybrid minus M2 BM25 | S2 | 128k | 200 | -4.0 [-10.0, +2.5] | no |
| M2 hybrid minus M2 BM25 | S2 | 256k | 200 | -1.5 [-8.0, +5.5] | no |
| M2 hybrid minus M2 BM25 | BABILong | 32k | 500 | -3.0 [-5.6, -0.4] | yes |
| M2 hybrid minus M2 BM25 | BABILong | 64k | 500 | +1.4 [-2.0, +4.8] | no |
| M2 hybrid minus M2 BM25 | BABILong | 128k | 500 | -3.4 [-7.0, +0.2] | no |
| M2 hybrid minus M2 BM25 | BABILong | 256k | 500 | -1.0 [-4.2, +2.2] | no |
| M2 hybrid minus M2 BM25 | BABILong-ext | 32k | 500 | -0.6 [-2.8, +1.6] | no |
| M2 hybrid minus M2 BM25 | BABILong-ext | 64k | 500 | -0.8 [-3.4, +1.8] | no |
| M2 hybrid minus M2 BM25 | BABILong-ext | 128k | 500 | -0.2 [-3.2, +2.6] | no |
| M2 hybrid minus M2 BM25 | BABILong-ext | 256k | 500 | -0.4 [-3.0, +2.2] | no |
| M2 hybrid minus M2 embeddings | S1 | 32k | 100 | -1.0 [-3.0, +0.0] | no |
| M2 hybrid minus M2 embeddings | S1 | 64k | 100 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M2 embeddings | S1 | 128k | 100 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M2 embeddings | S1 | 256k | 100 | +0.0 [+0.0, +0.0] | no |
| M2 hybrid minus M2 embeddings | S2 | 32k | 200 | -4.5 [-11.0, +2.0] | no |
| M2 hybrid minus M2 embeddings | S2 | 64k | 200 | +1.0 [-6.0, +8.0] | no |
| M2 hybrid minus M2 embeddings | S2 | 128k | 200 | -4.0 [-10.5, +2.5] | no |
| M2 hybrid minus M2 embeddings | S2 | 256k | 200 | +0.5 [-7.0, +8.0] | no |
| M2 hybrid minus M2 embeddings | BABILong | 32k | 500 | -0.4 [-2.6, +1.8] | no |
| M2 hybrid minus M2 embeddings | BABILong | 64k | 500 | +1.0 [-2.6, +4.4] | no |
| M2 hybrid minus M2 embeddings | BABILong | 128k | 500 | +0.8 [-2.8, +4.4] | no |
| M2 hybrid minus M2 embeddings | BABILong | 256k | 500 | +3.6 [+0.0, +7.2] | no |
| M2 hybrid minus M2 embeddings | BABILong-ext | 32k | 500 | -1.4 [-3.6, +0.8] | no |
| M2 hybrid minus M2 embeddings | BABILong-ext | 64k | 500 | +0.0 [-2.4, +2.4] | no |
| M2 hybrid minus M2 embeddings | BABILong-ext | 128k | 500 | -0.8 [-3.8, +2.2] | no |
| M2 hybrid minus M2 embeddings | BABILong-ext | 256k | 500 | +0.8 [-2.2, +3.8] | no |

## Evidence at budget (family means over tasks)

| Method | Family | Length | Item recall | Unit recall | Unit precision | Kept tokens | Accuracy, all evidence kept (n) | Accuracy, some evidence missing (n) | Recall at 8, 16, 32, 64, 128, 224 units |
|---|---|---|---|---|---|---|---|---|---|
| M2 BM25 | S1 | 32k | 1.0 | 1.0 | 0.0045 | 28614.3 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S1 | 64k | 1.0 | 1.0 | 0.0045 | 28606.7 | 99.0 [97.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S1 | 128k | 1.0 | 1.0 | 0.0046 | 28612.2 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S1 | 256k | 1.0 | 1.0 | 0.0046 | 28602.7 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S2 | 32k | 1.0 | 1.0 | 0.0112 | 28608.0 | 68.5 [62.0, 75.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S2 | 64k | 1.0 | 1.0 | 0.0113 | 28605.5 | 74.0 [68.0, 80.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S2 | 128k | 1.0 | 1.0 | 0.0114 | 28602.9 | 74.0 [68.0, 80.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | S2 | 256k | 1.0 | 1.0 | 0.0114 | 28602.4 | 79.0 [73.0, 84.5] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 BM25 | BABILong | 32k | 0.98 | 0.9792 | 0.0752 | 28536.7 | 47.6 [42.7, 52.7] (370) | 37.7 [29.2, 46.2] (130) | 0.415 / 0.474 / 0.539 / 0.624 / 0.783 / 0.980 |
| M2 BM25 | BABILong | 64k | 0.7391 | 0.7362 | 0.0519 | 28607.7 | 26.7 [19.3, 34.1] (135) | 42.5 [37.5, 47.7] (365) | 0.398 / 0.447 / 0.488 / 0.535 / 0.623 / 0.740 |
| M2 BM25 | BABILong | 128k | 0.6082 | 0.6066 | 0.0387 | 28609.4 | 21.6 [14.7, 29.3] (116) | 41.7 [36.7, 46.6] (384) | 0.364 / 0.415 / 0.458 / 0.498 / 0.544 / 0.609 |
| M2 BM25 | BABILong | 256k | 0.5343 | 0.5336 | 0.0318 | 28605.9 | 22.6 [15.1, 31.1] (106) | 37.1 [32.2, 41.9] (394) | 0.342 / 0.390 / 0.433 / 0.468 / 0.498 / 0.535 |
| M2 BM25 | BABILong-ext | 32k | 0.9812 | 0.9809 | 0.0314 | 28524.3 | 42.5 [37.9, 47.2] (428) | 34.7 [23.6, 45.8] (72) | 0.454 / 0.521 / 0.587 / 0.671 / 0.803 / 0.982 |
| M2 BM25 | BABILong-ext | 64k | 0.762 | 0.7609 | 0.0241 | 28602.9 | 50.0 [40.5, 58.6] (116) | 38.5 [33.9, 43.5] (384) | 0.407 / 0.474 / 0.534 / 0.589 / 0.665 / 0.765 |
| M2 BM25 | BABILong-ext | 128k | 0.6438 | 0.6439 | 0.02 | 28606.5 | 52.7 [40.5, 63.5] (74) | 39.0 [34.3, 43.7] (426) | 0.367 / 0.438 / 0.488 / 0.532 / 0.583 / 0.645 |
| M2 BM25 | BABILong-ext | 256k | 0.5819 | 0.5817 | 0.0181 | 28607.0 | 40.0 [26.7, 55.6] (45) | 43.5 [38.9, 48.1] (455) | 0.327 / 0.396 / 0.451 / 0.493 / 0.538 / 0.583 |
| M2 embeddings | S1 | 32k | 1.0 | 1.0 | 0.0045 | 28605.6 | 100.0 [100.0, 100.0] (100) | – (0) | 0.990 / 0.990 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 embeddings | S1 | 64k | 1.0 | 1.0 | 0.0045 | 28607.6 | 100.0 [100.0, 100.0] (100) | – (0) | 0.990 / 0.990 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 embeddings | S1 | 128k | 1.0 | 1.0 | 0.0045 | 28607.4 | 100.0 [100.0, 100.0] (100) | – (0) | 0.990 / 0.990 / 0.990 / 0.990 / 1.000 / 1.000 |
| M2 embeddings | S1 | 256k | 1.0 | 1.0 | 0.0045 | 28611.1 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 embeddings | S2 | 32k | 1.0 | 1.0 | 0.0112 | 28605.0 | 68.0 [61.5, 74.5] (200) | – (0) | 0.986 / 0.990 / 0.994 / 0.999 / 1.000 / 1.000 |
| M2 embeddings | S2 | 64k | 1.0 | 1.0 | 0.0112 | 28607.9 | 75.0 [69.0, 81.0] (200) | – (0) | 0.986 / 0.995 / 0.998 / 1.000 / 1.000 / 1.000 |
| M2 embeddings | S2 | 128k | 1.0 | 1.0 | 0.0112 | 28608.5 | 74.0 [68.0, 80.0] (200) | – (0) | 0.968 / 0.979 / 0.988 / 0.991 / 1.000 / 1.000 |
| M2 embeddings | S2 | 256k | 0.9963 | 0.9963 | 0.0112 | 28608.5 | 78.2 [72.1, 83.8] (197) | 0.0 [0.0, 0.0] (3) | 0.949 / 0.964 / 0.973 / 0.980 / 0.989 / 0.996 |
| M2 embeddings | BABILong | 32k | 0.9807 | 0.9802 | 0.0746 | 28532.6 | 44.4 [39.1, 49.4] (358) | 37.3 [29.6, 45.1] (142) | 0.299 / 0.375 / 0.474 / 0.605 / 0.797 / 0.981 |
| M2 embeddings | BABILong | 64k | 0.7565 | 0.7545 | 0.0533 | 28607.0 | 33.8 [25.6, 42.1] (133) | 40.3 [35.1, 45.5] (367) | 0.239 / 0.295 / 0.369 / 0.473 / 0.612 / 0.757 |
| M2 embeddings | BABILong | 128k | 0.5722 | 0.5713 | 0.0366 | 28609.2 | 27.0 [18.0, 36.0] (89) | 34.1 [29.7, 38.7] (411) | 0.188 / 0.237 / 0.295 / 0.368 / 0.471 / 0.572 |
| M2 embeddings | BABILong | 256k | 0.4592 | 0.4587 | 0.026 | 28604.9 | 18.5 [11.1, 27.2] (81) | 31.5 [27.2, 36.0] (419) | 0.153 / 0.196 / 0.244 / 0.305 / 0.379 / 0.459 |
| M2 embeddings | BABILong-ext | 32k | 0.9831 | 0.983 | 0.0314 | 28522.8 | 43.5 [38.9, 48.1] (437) | 33.3 [22.2, 46.0] (63) | 0.345 / 0.418 / 0.513 / 0.655 / 0.837 / 0.985 |
| M2 embeddings | BABILong-ext | 64k | 0.7917 | 0.7907 | 0.025 | 28605.9 | 40.7 [32.6, 48.9] (135) | 40.3 [35.3, 45.2] (365) | 0.300 / 0.351 / 0.418 / 0.521 / 0.656 / 0.792 |
| M2 embeddings | BABILong-ext | 128k | 0.5989 | 0.5988 | 0.019 | 28604.0 | 43.5 [30.4, 58.7] (46) | 41.4 [37.0, 45.8] (454) | 0.236 / 0.284 / 0.336 / 0.412 / 0.508 / 0.599 |
| M2 embeddings | BABILong-ext | 256k | 0.4837 | 0.4833 | 0.0151 | 28606.2 | 48.1 [29.6, 66.7] (27) | 41.6 [37.2, 46.1] (473) | 0.193 / 0.231 / 0.272 / 0.330 / 0.411 / 0.484 |
| M2 hybrid | S1 | 32k | 1.0 | 1.0 | 0.0045 | 28607.3 | 99.0 [97.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S1 | 64k | 1.0 | 1.0 | 0.0045 | 28604.5 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S1 | 128k | 1.0 | 1.0 | 0.0045 | 28614.0 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S1 | 256k | 1.0 | 1.0 | 0.0045 | 28606.6 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S2 | 32k | 1.0 | 1.0 | 0.0112 | 28611.8 | 63.5 [57.0, 70.0] (200) | – (0) | 0.999 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S2 | 64k | 1.0 | 1.0 | 0.0113 | 28607.2 | 76.0 [70.0, 82.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S2 | 128k | 1.0 | 1.0 | 0.0114 | 28605.7 | 70.0 [63.5, 76.5] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | S2 | 256k | 1.0 | 1.0 | 0.0114 | 28600.2 | 77.5 [71.5, 83.0] (200) | – (0) | 0.998 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid | BABILong | 32k | 0.981 | 0.9806 | 0.075 | 28532.5 | 44.6 [39.5, 49.5] (372) | 34.4 [26.6, 43.0] (128) | 0.406 / 0.479 / 0.555 / 0.658 / 0.814 / 0.982 |
| M2 hybrid | BABILong | 64k | 0.7731 | 0.7707 | 0.0549 | 28607.5 | 30.8 [23.3, 38.4] (146) | 43.2 [38.1, 48.6] (354) | 0.371 / 0.435 / 0.499 / 0.559 / 0.650 / 0.774 |
| M2 hybrid | BABILong | 128k | 0.6268 | 0.625 | 0.0405 | 28607.3 | 25.2 [17.9, 33.3] (123) | 36.3 [31.6, 41.1] (377) | 0.331 / 0.406 / 0.462 / 0.509 / 0.563 / 0.628 |
| M2 hybrid | BABILong | 256k | 0.5507 | 0.5499 | 0.0333 | 28607.6 | 19.0 [11.4, 26.7] (105) | 36.7 [32.2, 41.5] (395) | 0.294 / 0.371 / 0.428 / 0.472 / 0.508 / 0.551 |
| M2 hybrid | BABILong-ext | 32k | 0.9884 | 0.9881 | 0.0316 | 28527.9 | 41.3 [36.7, 45.9] (460) | 35.0 [20.0, 50.0] (40) | 0.458 / 0.535 / 0.614 / 0.710 / 0.857 / 0.990 |
| M2 hybrid | BABILong-ext | 64k | 0.8288 | 0.8277 | 0.0262 | 28607.7 | 48.4 [41.0, 55.9] (161) | 36.6 [31.6, 41.6] (339) | 0.417 / 0.492 / 0.548 / 0.620 / 0.717 / 0.830 |
| M2 hybrid | BABILong-ext | 128k | 0.6802 | 0.68 | 0.0212 | 28605.2 | 52.6 [42.3, 64.1] (78) | 38.6 [34.1, 43.4] (422) | 0.376 / 0.446 / 0.500 / 0.551 / 0.607 / 0.681 |
| M2 hybrid | BABILong-ext | 256k | 0.6078 | 0.6077 | 0.0189 | 28609.6 | 46.2 [32.7, 59.6] (52) | 42.4 [37.7, 47.1] (448) | 0.336 / 0.411 / 0.461 / 0.505 / 0.556 / 0.608 |
| M2 hybrid + reranker | S1 | 32k | 1.0 | 1.0 | 0.0045 | 28607.3 | 99.0 [97.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S1 | 64k | 1.0 | 1.0 | 0.0045 | 28604.5 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S1 | 128k | 1.0 | 1.0 | 0.0045 | 28614.0 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S1 | 256k | 1.0 | 1.0 | 0.0045 | 28606.6 | 100.0 [100.0, 100.0] (100) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S2 | 32k | 1.0 | 1.0 | 0.0112 | 28611.8 | 63.5 [57.0, 70.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S2 | 64k | 1.0 | 1.0 | 0.0113 | 28607.2 | 76.0 [70.0, 82.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S2 | 128k | 1.0 | 1.0 | 0.0114 | 28605.7 | 70.0 [63.5, 76.5] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | S2 | 256k | 1.0 | 1.0 | 0.0114 | 28600.2 | 77.5 [71.5, 83.0] (200) | – (0) | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000 |
| M2 hybrid + reranker | BABILong | 32k | 0.981 | 0.9806 | 0.075 | 28532.5 | 44.6 [39.5, 49.5] (372) | 34.4 [26.6, 43.0] (128) | 0.445 / 0.502 / 0.560 / 0.658 / 0.814 / 0.982 |
| M2 hybrid + reranker | BABILong | 64k | 0.7731 | 0.7707 | 0.0549 | 28607.5 | 30.8 [23.3, 38.4] (146) | 43.2 [38.1, 48.6] (354) | 0.420 / 0.466 / 0.509 / 0.559 / 0.650 / 0.774 |
| M2 hybrid + reranker | BABILong | 128k | 0.6268 | 0.625 | 0.0405 | 28607.3 | 25.2 [17.9, 33.3] (123) | 36.3 [31.6, 41.1] (377) | 0.399 / 0.444 / 0.479 / 0.509 / 0.563 / 0.628 |
| M2 hybrid + reranker | BABILong | 256k | 0.5507 | 0.5499 | 0.0333 | 28607.6 | 19.0 [11.4, 26.7] (105) | 36.7 [32.2, 41.5] (395) | 0.373 / 0.416 / 0.444 / 0.472 / 0.508 / 0.551 |
| M2 hybrid + reranker | BABILong-ext | 32k | 0.9884 | 0.9881 | 0.0316 | 28527.9 | 41.3 [36.7, 45.9] (460) | 35.0 [20.0, 50.0] (40) | 0.474 / 0.532 / 0.594 / 0.710 / 0.857 / 0.990 |
| M2 hybrid + reranker | BABILong-ext | 64k | 0.8288 | 0.8277 | 0.0262 | 28607.7 | 48.4 [41.0, 55.9] (161) | 36.6 [31.6, 41.6] (339) | 0.432 / 0.486 / 0.539 / 0.620 / 0.717 / 0.830 |
| M2 hybrid + reranker | BABILong-ext | 128k | 0.6802 | 0.68 | 0.0212 | 28605.2 | 52.6 [42.3, 64.1] (78) | 38.6 [34.1, 43.4] (422) | 0.397 / 0.455 / 0.502 / 0.551 / 0.607 / 0.681 |
| M2 hybrid + reranker | BABILong-ext | 256k | 0.6078 | 0.6077 | 0.0189 | 28609.6 | 46.2 [32.7, 59.6] (52) | 42.4 [37.7, 47.1] (448) | 0.364 / 0.419 / 0.463 / 0.505 / 0.556 / 0.608 |
| M3 oracle | S1 | 32k | 1.0 | 1.0 | 0.0045 | 28591.3 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle | S1 | 64k | 1.0 | 1.0 | 0.0045 | 28608.3 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle | S1 | 128k | 1.0 | 1.0 | 0.0045 | 28615.9 | 99.0 [97.0, 100.0] (100) | – (0) | – |
| M3 oracle | S1 | 256k | 1.0 | 1.0 | 0.0045 | 28605.0 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle | S2 | 32k | 1.0 | 1.0 | 0.0112 | 28597.9 | 69.5 [63.0, 75.5] (200) | – (0) | – |
| M3 oracle | S2 | 64k | 1.0 | 1.0 | 0.0112 | 28605.5 | 73.5 [67.0, 79.5] (200) | – (0) | – |
| M3 oracle | S2 | 128k | 1.0 | 1.0 | 0.0112 | 28600.9 | 79.0 [73.0, 84.5] (200) | – (0) | – |
| M3 oracle | S2 | 256k | 1.0 | 1.0 | 0.0112 | 28607.1 | 80.0 [74.0, 85.5] (200) | – (0) | – |
| M3 oracle | BABILong | 32k | 1.0 | 1.0 | 0.0771 | 28536.2 | 42.0 [37.6, 46.4] (500) | – (0) | – |
| M3 oracle | BABILong | 64k | 1.0 | 1.0 | 0.0809 | 28607.7 | 45.4 [41.0, 49.8] (500) | – (0) | – |
| M3 oracle | BABILong | 128k | 1.0 | 1.0 | 0.0829 | 28608.2 | 45.0 [40.6, 49.4] (500) | – (0) | – |
| M3 oracle | BABILong | 256k | 1.0 | 1.0 | 0.0841 | 28607.7 | 44.8 [40.4, 49.2] (500) | – (0) | – |
| M3 oracle | BABILong-ext | 32k | 1.0 | 1.0 | 0.032 | 28525.2 | 40.8 [36.6, 45.2] (500) | – (0) | – |
| M3 oracle | BABILong-ext | 64k | 1.0 | 1.0 | 0.0323 | 28607.9 | 41.0 [36.8, 45.4] (500) | – (0) | – |
| M3 oracle | BABILong-ext | 128k | 1.0 | 1.0 | 0.0326 | 28609.1 | 39.8 [35.4, 44.2] (500) | – (0) | – |
| M3 oracle | BABILong-ext | 256k | 1.0 | 1.0 | 0.0328 | 28607.3 | 40.2 [35.8, 44.6] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 8k | 1.0 | 1.0 | 0.3367 | 379.5 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 16k | 1.0 | 1.0 | 0.3333 | 383.6 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 32k | 1.0 | 1.0 | 0.3333 | 381.9 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 64k | 1.0 | 1.0 | 0.3333 | 381.9 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 128k | 1.0 | 1.0 | 0.3333 | 387.0 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S1 | 256k | 1.0 | 1.0 | 0.3333 | 384.7 | 100.0 [100.0, 100.0] (100) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 8k | 1.0 | 1.0 | 0.3413 | 905.5 | 95.0 [91.5, 98.0] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 16k | 1.0 | 1.0 | 0.3356 | 937.8 | 94.5 [91.0, 97.5] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 32k | 1.0 | 1.0 | 0.3359 | 945.0 | 95.0 [92.0, 97.5] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 64k | 1.0 | 1.0 | 0.3339 | 955.4 | 98.0 [96.0, 99.5] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 128k | 1.0 | 1.0 | 0.3337 | 958.6 | 97.0 [94.5, 99.0] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | S2 | 256k | 1.0 | 1.0 | 0.3343 | 953.1 | 97.5 [95.0, 99.5] (200) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 8k | 1.0 | 1.0 | 0.4415 | 3341.6 | 58.4 [54.0, 62.6] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 16k | 1.0 | 1.0 | 0.3891 | 4509.1 | 57.2 [52.8, 61.4] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 32k | 1.0 | 1.0 | 0.3627 | 5546.4 | 56.0 [51.6, 60.4] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 64k | 1.0 | 1.0 | 0.3487 | 6209.1 | 57.6 [53.2, 61.8] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 128k | 1.0 | 1.0 | 0.3441 | 6511.4 | 56.2 [51.8, 60.4] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong | 256k | 1.0 | 1.0 | 0.3417 | 6663.3 | 55.8 [51.4, 60.0] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 8k | 1.0 | 1.0 | 0.3799 | 2226.2 | 55.8 [51.6, 60.2] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 16k | 1.0 | 1.0 | 0.3542 | 2495.0 | 56.2 [51.8, 60.4] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 32k | 1.0 | 1.0 | 0.3441 | 2634.4 | 55.2 [50.8, 59.6] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 64k | 1.0 | 1.0 | 0.3392 | 2712.1 | 55.6 [51.2, 60.0] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 128k | 1.0 | 1.0 | 0.3366 | 2763.9 | 56.8 [52.4, 61.2] (500) | – (0) | – |
| M3 oracle, tight (supplementary) | BABILong-ext | 256k | 1.0 | 1.0 | 0.335 | 2795.8 | 55.0 [50.6, 59.4] (500) | – (0) | – |

## Cost (means over the cells of a method at a length)

| Method | Length | Cells | Rows ok | Reused answers | Prompt tokens | s per example | First token s | Selection ms | Rerank ms | CPU preparation s | Peak VRAM GiB | Ingest: documents, tokens/s, s per document |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 dense YaRN | 64k | 13 | 470 | 0 | 62935.0 | 21.077 | 19.919 | None | None | 0.206 | 11.365 | 0, None, None |
| M1 dense YaRN | 128k | 13 | 470 | 0 | 125888.9 | 74.896 | 71.792 | None | None | 0.431 | 19.384 | 0, None, None |
| M1 dense YaRN | full | 2 | 60 | 0 | 125368.5 | 82.802 | 70.722 | None | None | 0.355 | 18.32 | 0, None, None |
| M2 BM25 | 32k | 13 | 1300 | 65 | 28793.5 | 5.736 | 5.249 | 32.43 | None | 0.153 | 7.85 | 0, None, None |
| M2 BM25 | 64k | 13 | 1300 | 0 | 29203.4 | 5.85 | 5.347 | 64.66 | None | 0.316 | 7.914 | 0, None, None |
| M2 BM25 | 128k | 13 | 1300 | 0 | 29400.9 | 5.981 | 5.476 | 117.01 | None | 0.61 | 7.916 | 0, None, None |
| M2 BM25 | 256k | 13 | 1300 | 5 | 29506.4 | 6.162 | 5.611 | 202.16 | None | 1.17 | 7.924 | 0, None, None |
| M2 BM25 | full | 4 | 400 | 124 | 29309.0 | 7.181 | 5.459 | 96.86 | None | 0.611 | 7.917 | 0, None, None |
| M2 embeddings | 32k | 13 | 1300 | 68 | 28792.2 | 5.817 | 5.244 | 65.99 | None | 0.143 | 7.85 | 0, None, None |
| M2 embeddings | 64k | 13 | 1300 | 1 | 29145.4 | 5.908 | 5.324 | 66.34 | None | 0.269 | 7.914 | 0, None, None |
| M2 embeddings | 128k | 13 | 1300 | 0 | 29302.0 | 5.933 | 5.365 | 66.46 | None | 0.533 | 7.915 | 0, None, None |
| M2 embeddings | 256k | 13 | 1300 | 5 | 29402.0 | 6.026 | 5.411 | 69.29 | None | 1.017 | 7.917 | 0, None, None |
| M2 embeddings | full | 4 | 400 | 94 | 29428.2 | 6.94 | 5.417 | 72.91 | None | 0.531 | 7.917 | 0, None, None |
| M2 hybrid | 32k | 13 | 1300 | 5 | 28799.8 | 5.947 | 5.357 | 112.82 | None | 0.191 | 9.021 | 1290, 29418.2, 1.059 |
| M2 hybrid | 64k | 13 | 1300 | 0 | 29199.7 | 5.956 | 5.402 | 129.42 | None | 0.326 | 9.019 | 1300, 32619.7, 1.921 |
| M2 hybrid | 128k | 13 | 1300 | 0 | 29378.1 | 6.089 | 5.472 | 182.91 | None | 0.652 | 9.024 | 1300, 32785.6, 3.835 |
| M2 hybrid | 256k | 13 | 1300 | 5 | 29478.6 | 6.115 | 5.497 | 269.68 | None | 1.255 | 9.028 | 1290, 33646.8, 7.486 |
| M2 hybrid | full | 4 | 400 | 5 | 29482.5 | 7.085 | 5.52 | 170.96 | None | 0.67 | 9.027 | 394, 41708.9, 3.707 |
| M2 hybrid + reranker | 32k | 13 | 1300 | 1300 | 28799.8 | None | None | None | 479.17 | None | 6.68 | 0, None, None |
| M2 hybrid + reranker | 64k | 13 | 1300 | 1300 | 29199.7 | None | None | None | 476.9 | None | 6.681 | 0, None, None |
| M2 hybrid + reranker | 128k | 13 | 1300 | 1300 | 29378.1 | None | None | None | 502.03 | None | 6.664 | 0, None, None |
| M2 hybrid + reranker | 256k | 13 | 1300 | 1300 | 29478.6 | None | None | None | 466.9 | None | 6.676 | 0, None, None |
| M2 hybrid + reranker | full | 4 | 400 | 400 | 29482.5 | None | None | None | 523.14 | None | 6.81 | 0, None, None |
| M3 oracle | 32k | 13 | 1300 | 0 | 28733.0 | 5.724 | 5.232 | 0.57 | None | 0.127 | 6.814 | 0, None, None |
| M3 oracle | 64k | 13 | 1300 | 12 | 28810.6 | 5.729 | 5.252 | 0.74 | None | 0.261 | 6.814 | 0, None, None |
| M3 oracle | 128k | 13 | 1300 | 17 | 28820.3 | 5.879 | 5.285 | 1.26 | None | 0.628 | 6.814 | 0, None, None |
| M3 oracle | 256k | 13 | 1300 | 26 | 28824.3 | 5.705 | 5.241 | 1.64 | None | 0.978 | 6.814 | 0, None, None |
| M3 oracle | full | 3 | 300 | 5 | 28706.3 | 7.398 | 5.218 | 0.76 | None | 0.434 | 6.814 | 0, None, None |
| M3 oracle, tight (supplementary) | 8k | 13 | 1300 | 5 | 2504.8 | 0.617 | 0.221 | 0.32 | None | 0.03 | 4.26 | 0, None, None |
| M3 oracle, tight (supplementary) | 16k | 13 | 1300 | 0 | 3069.0 | 0.691 | 0.293 | 0.38 | None | 0.059 | 5.085 | 0, None, None |
| M3 oracle, tight (supplementary) | 32k | 13 | 1300 | 0 | 3528.9 | 0.775 | 0.367 | 0.45 | None | 0.12 | 6.558 | 0, None, None |
| M3 oracle, tight (supplementary) | 64k | 13 | 1300 | 0 | 3820.9 | 0.847 | 0.425 | 0.59 | None | 0.249 | 6.808 | 0, None, None |
| M3 oracle, tight (supplementary) | 128k | 13 | 1300 | 0 | 3962.2 | 0.893 | 0.451 | 0.97 | None | 0.496 | 6.808 | 0, None, None |
| M3 oracle, tight (supplementary) | 256k | 13 | 1300 | 5 | 4034.2 | 0.909 | 0.472 | 1.57 | None | 0.966 | 6.808 | 0, None, None |
| M3 oracle, tight (supplementary) | full | 3 | 300 | 5 | 480.2 | 1.93 | 0.079 | 0.62 | None | 0.449 | 3.36 | 0, None, None |

## InfiniteBench (accuracy %, 95% CI; evidence recall)

| Method | passkey | number_string | kv_retrieval | longbook_choice_eng |
|---|---|---|---|---|
| M0 last 32K | 27.0 [19.0, 36.0] (n=100, recall 0.27) | 27.0 [19.0, 36.0] (n=100, recall 0.27) | 6.0 [2.0, 11.0] (n=100, recall 0.19) | – |
| M0 first + last 16K | 27.0 [19.0, 36.0] (n=100, recall 0.27) | 27.0 [19.0, 36.0] (n=100, recall 0.27) | 3.0 [0.0, 7.0] (n=100, recall 0.19) | – |
| M1 dense YaRN | 13.3 [3.3, 26.7] (n=30, recall 1.0) | 40.0 [23.3, 56.7] (n=30, recall 1.0) | – | – |
| M2 BM25 | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 33.0 [24.0, 42.0] (n=100, recall 0.69) | 49.0 [39.0, 59.0] (n=100) |
| M2 embeddings | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 17.0 [10.0, 25.0] (n=100, recall 0.51) | 49.0 [39.0, 59.0] (n=100) |
| M2 hybrid | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 18.0 [11.0, 26.0] (n=100, recall 0.52) | 51.0 [41.0, 61.0] (n=100) |
| M2 hybrid + reranker | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 18.0 [11.0, 26.0] (n=100, recall 0.52) | 51.0 [41.0, 61.0] (n=100) |
| M3 oracle | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 21.0 [13.0, 29.0] (n=100, recall 1.0) | – |
| M3 oracle, tight (supplementary) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 100.0 [100.0, 100.0] (n=100, recall 1.0) | 55.0 [45.0, 65.0] (n=100, recall 1.0) | – |

## Every cell of E2, E3 and E3o

| Method | Task | Length | Status | Accuracy | Whole-word (BABILong) | Evidence recall | s per example | Peak VRAM GiB | Error kinds |
|---|---|---|---|---|---|---|---|---|---|
| native-1.7B-a4budget | synth:S2_multivalue | 8k | {'ok': 100} | 79.0 [71.0, 87.0] | – | 1.0 | 2.895 | 4.259 | {'correct': 79, 'wrong': 20, 'format': 1} |
| native-1.7B-a4budget | synth:S2_multivalue | 16k | {'ok': 100} | 72.0 [63.0, 81.0] | – | 1.0 | 4.469 | 5.242 | {'correct': 72, 'wrong': 28} |
| native-1.7B-a4budget | synth:S2_multivalue | 32k | {'ok': 100} | 59.0 [49.0, 68.0] | – | 1.0 | 9.461 | 7.285 | {'correct': 59, 'wrong': 41} |
| m1-yarn-1.7B | babilong:qa1 | 64k | {'ok': 40} | 55.0 [40.0, 70.0] | 55.0 [40.0, 70.0] | 1.0 | 20.826 | 11.119 | {'correct': 22, 'wrong': 12, 'format': 6} |
| m1-yarn-1.7B | babilong:qa1 | 128k | {'ok': 40} | 17.5 [7.5, 30.0] | 17.5 [7.5, 30.0] | 1.0 | 72.556 | 18.943 | {'format': 29, 'correct': 7, 'wrong': 4} |
| m1-yarn-1.7B | babilong:qa10 | 64k | {'ok': 30} | 43.3 [26.7, 60.0] | 43.3 [26.7, 60.0] | 1.0 | 19.656 | 11.095 | {'correct': 13, 'wrong': 17} |
| m1-yarn-1.7B | babilong:qa10 | 128k | {'ok': 30} | 40.0 [23.3, 56.7] | 40.0 [23.3, 56.7] | 1.0 | 70.871 | 18.792 | {'correct': 12, 'wrong': 17, 'format': 1} |
| m1-yarn-1.7B | babilong:qa2 | 64k | {'ok': 40} | 27.5 [15.0, 42.5] | 27.5 [15.0, 42.5] | 1.0 | 20.518 | 11.095 | {'wrong': 23, 'correct': 11, 'format': 6} |
| m1-yarn-1.7B | babilong:qa2 | 128k | {'ok': 40} | 17.5 [7.5, 30.0] | 17.5 [7.5, 30.0] | 1.0 | 72.64 | 18.909 | {'wrong': 16, 'format': 17, 'correct': 7} |
| m1-yarn-1.7B | babilong:qa3 | 64k | {'ok': 40} | 40.0 [25.0, 55.0] | 40.0 [25.0, 55.0] | 1.0 | 20.889 | 11.261 | {'correct': 16, 'wrong': 24} |
| m1-yarn-1.7B | babilong:qa3 | 128k | {'ok': 40} | 27.5 [15.0, 42.5] | 27.5 [15.0, 42.5] | 1.0 | 72.506 | 18.959 | {'wrong': 29, 'correct': 11} |
| m1-yarn-1.7B | babilong:qa4 | 64k | {'ok': 40} | 30.0 [15.0, 45.0] | 30.0 [15.0, 45.0] | 1.0 | 20.276 | 11.042 | {'wrong': 28, 'correct': 12} |
| m1-yarn-1.7B | babilong:qa4 | 128k | {'ok': 40} | 20.0 [7.5, 32.5] | 20.0 [7.5, 32.5] | 1.0 | 72.309 | 18.883 | {'wrong': 28, 'correct': 8, 'format': 4} |
| m1-yarn-1.7B | babilong:qa5 | 64k | {'ok': 40} | 67.5 [52.5, 82.5] | 67.5 [52.5, 82.5] | 1.0 | 19.605 | 11.058 | {'correct': 27, 'wrong': 13} |
| m1-yarn-1.7B | babilong:qa5 | 128k | {'ok': 40} | 47.5 [32.5, 62.5] | 47.5 [32.5, 62.5] | 1.0 | 71.582 | 18.987 | {'correct': 19, 'format': 3, 'wrong': 18} |
| m1-yarn-1.7B | babilong:qa6 | 64k | {'ok': 30} | 66.7 [50.0, 83.3] | 66.7 [50.0, 83.3] | 1.0 | 19.521 | 11.109 | {'correct': 20, 'wrong': 10} |
| m1-yarn-1.7B | babilong:qa6 | 128k | {'ok': 30} | 60.0 [43.3, 76.7] | 60.0 [43.3, 76.7] | 1.0 | 70.998 | 18.782 | {'correct': 18, 'wrong': 12} |
| m1-yarn-1.7B | babilong:qa7 | 64k | {'ok': 30} | 0.0 [0.0, 0.0] | 6.7 [0.0, 16.7] | 1.0 | 19.619 | 11.297 | {'format': 26, 'wrong': 4} |
| m1-yarn-1.7B | babilong:qa7 | 128k | {'ok': 30} | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1.0 | 70.715 | 18.838 | {'format': 29, 'wrong': 1} |
| m1-yarn-1.7B | babilong:qa8 | 64k | {'ok': 30} | 26.7 [13.3, 43.3] | 26.7 [13.3, 43.3] | 1.0 | 20.109 | 11.259 | {'format': 14, 'wrong': 8, 'correct': 8} |
| m1-yarn-1.7B | babilong:qa8 | 128k | {'ok': 30} | 6.7 [0.0, 16.7] | 6.7 [0.0, 16.7] | 1.0 | 71.738 | 18.988 | {'format': 19, 'wrong': 9, 'correct': 2} |
| m1-yarn-1.7B | babilong:qa9 | 64k | {'ok': 30} | 56.7 [40.0, 73.3] | 56.7 [40.0, 73.3] | 1.0 | 19.433 | 11.042 | {'wrong': 13, 'correct': 17} |
| m1-yarn-1.7B | babilong:qa9 | 128k | {'ok': 30} | 53.3 [36.7, 70.0] | 53.3 [36.7, 70.0] | 1.0 | 70.344 | 18.979 | {'correct': 16, 'wrong': 13, 'format': 1} |
| m1-yarn-1.7B | infbench:number_string | full | {'ok': 30} | 40.0 [23.3, 56.7] | – | 1.0 | 82.499 | 18.32 | {'correct': 12, 'format': 17, 'wrong': 1} |
| m1-yarn-1.7B | infbench:passkey | full | {'ok': 30} | 13.3 [3.3, 26.7] | – | 1.0 | 83.106 | 18.319 | {'correct': 4, 'format': 26} |
| m1-yarn-1.7B | synth:S1 | 64k | {'ok': 40} | 100.0 [100.0, 100.0] | – | 1.0 | 22.415 | 11.357 | {'correct': 40} |
| m1-yarn-1.7B | synth:S1 | 128k | {'ok': 40} | 100.0 [100.0, 100.0] | – | 1.0 | 82.051 | 19.381 | {'correct': 40} |
| m1-yarn-1.7B | synth:S2_multikey | 64k | {'ok': 40} | 52.5 [37.5, 67.5] | – | 1.0 | 22.393 | 11.358 | {'wrong': 19, 'correct': 21} |
| m1-yarn-1.7B | synth:S2_multikey | 128k | {'ok': 40} | 35.0 [20.0, 50.0] | – | 1.0 | 81.422 | 19.383 | {'correct': 14, 'wrong': 26} |
| m1-yarn-1.7B | synth:S2_multivalue | 64k | {'ok': 40} | 72.5 [57.5, 85.0] | – | 1.0 | 28.743 | 11.365 | {'correct': 29, 'wrong': 11} |
| m1-yarn-1.7B | synth:S2_multivalue | 128k | {'ok': 40} | 27.5 [15.0, 42.5] | – | 1.0 | 93.911 | 19.384 | {'wrong': 27, 'correct': 11, 'format': 2} |
| m2-bm25-1.7B | babilong:qa1 | 32k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 0.966 | 5.867 | 7.837 | {'format': 9, 'correct': 70, 'wrong': 21} |
| m2-bm25-1.7B | babilong:qa1 | 64k | {'ok': 100} | 58.0 [48.0, 67.0] | 58.0 [48.0, 67.0] | 0.7047 | 5.971 | 7.879 | {'correct': 58, 'wrong': 24, 'format': 18} |
| m2-bm25-1.7B | babilong:qa1 | 128k | {'ok': 100} | 52.0 [42.0, 62.0] | 52.0 [42.0, 62.0] | 0.5573 | 6.111 | 7.904 | {'format': 26, 'correct': 52, 'wrong': 22} |
| m2-bm25-1.7B | babilong:qa1 | 256k | {'ok': 100} | 55.0 [45.0, 64.0] | 55.0 [45.0, 64.0] | 0.5257 | 6.249 | 7.918 | {'correct': 55, 'format': 25, 'wrong': 20} |
| m2-bm25-1.7B | babilong:qa10 | 32k | {'ok': 100} | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] | 0.987 | 5.314 | 7.843 | {'wrong': 45, 'correct': 55} |
| m2-bm25-1.7B | babilong:qa10 | 64k | {'ok': 100} | 47.0 [37.0, 57.0] | 47.0 [37.0, 57.0] | 0.8411 | 5.418 | 7.914 | {'wrong': 53, 'correct': 47} |
| m2-bm25-1.7B | babilong:qa10 | 128k | {'ok': 100} | 51.0 [41.0, 61.0] | 51.0 [41.0, 61.0] | 0.7364 | 5.551 | 7.915 | {'correct': 51, 'wrong': 49} |
| m2-bm25-1.7B | babilong:qa10 | 256k | {'ok': 100} | 52.0 [42.0, 62.0] | 52.0 [42.0, 62.0] | 0.6378 | 5.684 | 7.917 | {'correct': 52, 'wrong': 48} |
| m2-bm25-1.7B | babilong:qa2 | 32k | {'ok': 100} | 32.0 [23.0, 41.0] | 32.0 [23.0, 41.0] | 0.974 | 5.694 | 7.844 | {'correct': 32, 'wrong': 64, 'format': 4} |
| m2-bm25-1.7B | babilong:qa2 | 64k | {'ok': 100} | 35.0 [26.0, 44.0] | 35.0 [26.0, 44.0] | 0.62 | 5.81 | 7.883 | {'wrong': 54, 'correct': 35, 'format': 11} |
| m2-bm25-1.7B | babilong:qa2 | 128k | {'ok': 100} | 20.0 [12.0, 28.0] | 20.0 [12.0, 28.0] | 0.4369 | 5.945 | 7.904 | {'format': 29, 'wrong': 51, 'correct': 20} |
| m2-bm25-1.7B | babilong:qa2 | 256k | {'ok': 100} | 13.0 [7.0, 20.0] | 13.0 [7.0, 20.0] | 0.2914 | 6.081 | 7.918 | {'correct': 13, 'wrong': 53, 'format': 34} |
| m2-bm25-1.7B | babilong:qa3 | 32k | {'ok': 100} | 34.0 [25.0, 44.0] | 34.0 [25.0, 44.0] | 0.9763 | 5.901 | 7.85 | {'wrong': 66, 'correct': 34} |
| m2-bm25-1.7B | babilong:qa3 | 64k | {'ok': 100} | 24.0 [16.0, 33.0] | 24.0 [16.0, 33.0] | 0.6228 | 6.006 | 7.894 | {'wrong': 76, 'correct': 24} |
| m2-bm25-1.7B | babilong:qa3 | 128k | {'ok': 100} | 29.0 [20.0, 38.0] | 29.0 [20.0, 38.0] | 0.4362 | 6.127 | 7.913 | {'wrong': 71, 'correct': 29} |
| m2-bm25-1.7B | babilong:qa3 | 256k | {'ok': 100} | 21.0 [13.0, 29.0] | 21.0 [13.0, 29.0] | 0.3532 | 6.288 | 7.924 | {'wrong': 79, 'correct': 21} |
| m2-bm25-1.7B | babilong:qa4 | 32k | {'ok': 100} | 22.0 [14.0, 30.0] | 22.0 [14.0, 30.0] | 1.0 | 5.434 | 7.837 | {'wrong': 74, 'correct': 22, 'format': 4} |
| m2-bm25-1.7B | babilong:qa4 | 64k | {'ok': 100} | 17.0 [10.0, 25.0] | 17.0 [10.0, 25.0] | 0.995 | 5.499 | 7.882 | {'wrong': 80, 'correct': 17, 'format': 3} |
| m2-bm25-1.7B | babilong:qa4 | 128k | {'ok': 100} | 15.0 [8.0, 22.0] | 15.0 [8.0, 22.0] | 0.99 | 5.602 | 7.9 | {'wrong': 83, 'correct': 15, 'format': 2} |
| m2-bm25-1.7B | babilong:qa4 | 256k | {'ok': 100} | 16.0 [9.0, 23.0] | 16.0 [9.0, 23.0] | 0.95 | 5.734 | 7.909 | {'wrong': 82, 'correct': 16, 'format': 2} |
| m2-bm25-1.7B | babilong:qa5 | 32k | {'ok': 100} | 67.0 [57.0, 76.0] | 67.0 [57.0, 76.0] | 0.9836 | 5.335 | 7.843 | {'correct': 67, 'wrong': 33} |
| m2-bm25-1.7B | babilong:qa5 | 64k | {'ok': 100} | 57.0 [47.0, 67.0] | 57.0 [47.0, 67.0] | 0.7529 | 5.446 | 7.886 | {'correct': 57, 'wrong': 43} |
| m2-bm25-1.7B | babilong:qa5 | 128k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 0.6206 | 5.571 | 7.906 | {'correct': 69, 'wrong': 31} |
| m2-bm25-1.7B | babilong:qa5 | 256k | {'ok': 100} | 65.0 [55.0, 74.0] | 65.0 [55.0, 74.0] | 0.5513 | 5.689 | 7.919 | {'correct': 65, 'wrong': 35} |
| m2-bm25-1.7B | babilong:qa6 | 32k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 0.9808 | 5.313 | 7.841 | {'correct': 63, 'wrong': 37} |
| m2-bm25-1.7B | babilong:qa6 | 64k | {'ok': 100} | 66.0 [56.0, 75.0] | 66.0 [56.0, 75.0] | 0.7125 | 5.412 | 7.913 | {'correct': 66, 'wrong': 34} |
| m2-bm25-1.7B | babilong:qa6 | 128k | {'ok': 100} | 58.0 [48.0, 68.0] | 58.0 [48.0, 68.0] | 0.5952 | 5.543 | 7.915 | {'correct': 58, 'wrong': 42} |
| m2-bm25-1.7B | babilong:qa6 | 256k | {'ok': 100} | 67.0 [57.0, 76.0] | 67.0 [57.0, 76.0] | 0.5237 | 5.674 | 7.916 | {'correct': 67, 'wrong': 33} |
| m2-bm25-1.7B | babilong:qa7 | 32k | {'ok': 100} | 0.0 [0.0, 0.0] | 10.0 [5.0, 16.0] | 0.9736 | 5.404 | 7.839 | {'format': 77, 'wrong': 23} |
| m2-bm25-1.7B | babilong:qa7 | 64k | {'ok': 100} | 0.0 [0.0, 0.0] | 20.0 [12.0, 28.0] | 0.706 | 5.476 | 7.914 | {'wrong': 45, 'format': 55} |
| m2-bm25-1.7B | babilong:qa7 | 128k | {'ok': 100} | 0.0 [0.0, 0.0] | 29.0 [20.0, 38.0] | 0.5981 | 5.603 | 7.915 | {'wrong': 61, 'format': 39} |
| m2-bm25-1.7B | babilong:qa7 | 256k | {'ok': 100} | 0.0 [0.0, 0.0] | 27.0 [18.0, 36.0] | 0.5437 | 5.735 | 7.918 | {'wrong': 53, 'format': 47} |
| m2-bm25-1.7B | babilong:qa8 | 32k | {'ok': 100} | 20.0 [13.0, 28.0] | 20.0 [13.0, 28.0] | 0.9783 | 5.487 | 7.841 | {'format': 35, 'correct': 20, 'wrong': 45} |
| m2-bm25-1.7B | babilong:qa8 | 64k | {'ok': 100} | 23.0 [15.0, 31.0] | 23.0 [15.0, 31.0] | 0.7145 | 5.579 | 7.913 | {'wrong': 42, 'format': 35, 'correct': 23} |
| m2-bm25-1.7B | babilong:qa8 | 128k | {'ok': 100} | 26.0 [18.0, 35.0] | 26.0 [18.0, 35.0] | 0.5853 | 5.712 | 7.916 | {'correct': 26, 'wrong': 49, 'format': 25} |
| m2-bm25-1.7B | babilong:qa8 | 256k | {'ok': 100} | 28.0 [19.0, 37.0] | 28.0 [19.0, 37.0] | 0.5368 | 5.842 | 7.92 | {'correct': 28, 'wrong': 53, 'format': 19} |
| m2-bm25-1.7B | babilong:qa9 | 32k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 0.9863 | 5.313 | 7.84 | {'correct': 69, 'wrong': 31} |
| m2-bm25-1.7B | babilong:qa9 | 64k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 0.8358 | 5.408 | 7.912 | {'correct': 70, 'wrong': 30} |
| m2-bm25-1.7B | babilong:qa9 | 128k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 0.7041 | 5.533 | 7.915 | {'correct': 70, 'wrong': 30} |
| m2-bm25-1.7B | babilong:qa9 | 256k | {'ok': 100} | 69.0 [59.0, 78.0] | 69.0 [59.0, 78.0] | 0.6673 | 5.682 | 7.916 | {'correct': 69, 'wrong': 31} |
| m2-bm25-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 33.0 [24.0, 42.0] | – | 0.69 | 9.804 | 7.835 | {'correct': 33, 'wrong': 67} |
| m2-bm25-1.7B | infbench:longbook_choice_eng | full | {'ok': 100} | 49.0 [39.0, 59.0] | – | None | 6.051 | 7.916 | {'wrong': 51, 'correct': 49} |
| m2-bm25-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.594 | 7.917 | {'correct': 100} |
| m2-bm25-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.273 | 7.913 | {'correct': 100} |
| m2-bm25-1.7B | synth:S1 | 32k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.682 | 7.835 | {'correct': 100} |
| m2-bm25-1.7B | synth:S1 | 64k | {'ok': 100} | 99.0 [97.0, 100.0] | – | 1.0 | 5.777 | 7.877 | {'correct': 99, 'format': 1} |
| m2-bm25-1.7B | synth:S1 | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.899 | 7.895 | {'correct': 100} |
| m2-bm25-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.047 | 7.917 | {'correct': 100} |
| m2-bm25-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 77.0 [68.0, 85.0] | – | 1.0 | 5.679 | 7.836 | {'correct': 77, 'wrong': 23} |
| m2-bm25-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 81.0 [73.0, 88.0] | – | 1.0 | 5.778 | 7.878 | {'correct': 81, 'wrong': 19} |
| m2-bm25-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 81.0 [73.0, 88.0] | – | 1.0 | 5.895 | 7.893 | {'correct': 81, 'wrong': 19} |
| m2-bm25-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 81.0 [73.0, 88.0] | – | 1.0 | 6.055 | 7.917 | {'correct': 81, 'wrong': 19} |
| m2-bm25-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 60.0 [50.0, 69.0] | – | 1.0 | 8.139 | 7.842 | {'correct': 60, 'wrong': 40} |
| m2-bm25-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 67.0 [58.0, 76.0] | – | 1.0 | 8.467 | 7.885 | {'correct': 67, 'wrong': 33} |
| m2-bm25-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 67.0 [58.0, 76.0] | – | 1.0 | 8.664 | 7.903 | {'wrong': 31, 'format': 2, 'correct': 67} |
| m2-bm25-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 77.0 [68.0, 85.0] | – | 1.0 | 9.345 | 7.917 | {'correct': 77, 'wrong': 21, 'format': 2} |
| m2-embed-1.7B | babilong:qa1 | 32k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 0.987 | 5.93 | 7.841 | {'format': 10, 'correct': 63, 'wrong': 27} |
| m2-embed-1.7B | babilong:qa1 | 64k | {'ok': 100} | 60.0 [50.0, 69.0] | 60.0 [50.0, 69.0] | 0.808 | 6.009 | 7.912 | {'correct': 60, 'format': 15, 'wrong': 25} |
| m2-embed-1.7B | babilong:qa1 | 128k | {'ok': 100} | 43.0 [33.0, 53.0] | 43.0 [33.0, 53.0] | 0.6118 | 6.08 | 7.914 | {'format': 30, 'correct': 43, 'wrong': 27} |
| m2-embed-1.7B | babilong:qa1 | 256k | {'ok': 100} | 40.0 [30.0, 50.0] | 40.0 [30.0, 50.0] | 0.5077 | 6.129 | 7.911 | {'correct': 40, 'format': 43, 'wrong': 17} |
| m2-embed-1.7B | babilong:qa10 | 32k | {'ok': 100} | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] | 0.9829 | 5.372 | 7.843 | {'wrong': 45, 'correct': 55} |
| m2-embed-1.7B | babilong:qa10 | 64k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 0.8092 | 5.454 | 7.912 | {'correct': 45, 'wrong': 55} |
| m2-embed-1.7B | babilong:qa10 | 128k | {'ok': 100} | 46.0 [36.0, 56.0] | 46.0 [36.0, 56.0] | 0.6163 | 5.496 | 7.915 | {'wrong': 54, 'correct': 46} |
| m2-embed-1.7B | babilong:qa10 | 256k | {'ok': 100} | 50.0 [40.0, 60.0] | 50.0 [40.0, 60.0] | 0.5022 | 5.538 | 7.916 | {'correct': 50, 'wrong': 50} |
| m2-embed-1.7B | babilong:qa2 | 32k | {'ok': 100} | 26.0 [18.0, 35.0] | 26.0 [18.0, 35.0] | 0.9644 | 5.759 | 7.842 | {'correct': 26, 'wrong': 71, 'format': 3} |
| m2-embed-1.7B | babilong:qa2 | 64k | {'ok': 100} | 26.0 [18.0, 35.0] | 26.0 [18.0, 35.0] | 0.6348 | 5.856 | 7.913 | {'correct': 26, 'format': 22, 'wrong': 52} |
| m2-embed-1.7B | babilong:qa2 | 128k | {'ok': 100} | 14.0 [8.0, 21.0] | 14.0 [8.0, 21.0] | 0.4087 | 5.926 | 7.915 | {'format': 28, 'wrong': 58, 'correct': 14} |
| m2-embed-1.7B | babilong:qa2 | 256k | {'ok': 100} | 13.0 [7.0, 20.0] | 13.0 [7.0, 20.0] | 0.2711 | 5.954 | 7.915 | {'format': 23, 'wrong': 64, 'correct': 13} |
| m2-embed-1.7B | babilong:qa3 | 32k | {'ok': 100} | 36.0 [27.0, 46.0] | 36.0 [27.0, 46.0] | 0.9696 | 5.975 | 7.85 | {'wrong': 64, 'correct': 36} |
| m2-embed-1.7B | babilong:qa3 | 64k | {'ok': 100} | 28.0 [20.0, 37.0] | 28.0 [20.0, 37.0] | 0.6367 | 6.041 | 7.914 | {'wrong': 72, 'correct': 28} |
| m2-embed-1.7B | babilong:qa3 | 128k | {'ok': 100} | 24.0 [16.0, 33.0] | 24.0 [16.0, 33.0] | 0.3969 | 6.107 | 7.914 | {'wrong': 76, 'correct': 24} |
| m2-embed-1.7B | babilong:qa3 | 256k | {'ok': 100} | 23.0 [15.0, 32.0] | 23.0 [15.0, 32.0] | 0.2517 | 6.146 | 7.917 | {'correct': 23, 'wrong': 77} |
| m2-embed-1.7B | babilong:qa4 | 32k | {'ok': 100} | 25.0 [17.0, 34.0] | 25.0 [17.0, 34.0] | 1.0 | 5.487 | 7.838 | {'wrong': 72, 'correct': 25, 'format': 3} |
| m2-embed-1.7B | babilong:qa4 | 64k | {'ok': 100} | 21.0 [13.0, 29.0] | 21.0 [13.0, 29.0] | 0.965 | 5.545 | 7.913 | {'correct': 21, 'wrong': 76, 'format': 3} |
| m2-embed-1.7B | babilong:qa4 | 128k | {'ok': 100} | 19.0 [12.0, 27.0] | 19.0 [12.0, 27.0] | 0.875 | 5.596 | 7.914 | {'wrong': 78, 'correct': 19, 'format': 3} |
| m2-embed-1.7B | babilong:qa4 | 256k | {'ok': 100} | 17.0 [10.0, 25.0] | 17.0 [10.0, 25.0] | 0.79 | 5.652 | 7.916 | {'wrong': 83, 'correct': 17} |
| m2-embed-1.7B | babilong:qa5 | 32k | {'ok': 100} | 62.0 [52.0, 71.0] | 62.0 [52.0, 71.0] | 0.9827 | 5.399 | 7.842 | {'correct': 62, 'wrong': 38} |
| m2-embed-1.7B | babilong:qa5 | 64k | {'ok': 100} | 58.0 [48.0, 68.0] | 58.0 [48.0, 68.0] | 0.7379 | 5.472 | 7.912 | {'correct': 58, 'wrong': 42} |
| m2-embed-1.7B | babilong:qa5 | 128k | {'ok': 100} | 64.0 [55.0, 73.0] | 64.0 [55.0, 73.0] | 0.5686 | 5.52 | 7.915 | {'correct': 64, 'wrong': 36} |
| m2-embed-1.7B | babilong:qa5 | 256k | {'ok': 100} | 54.0 [44.0, 64.0] | 54.0 [44.0, 64.0] | 0.4756 | 5.568 | 7.916 | {'correct': 54, 'wrong': 46} |
| m2-embed-1.7B | babilong:qa6 | 32k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 0.9865 | 5.369 | 7.839 | {'correct': 63, 'wrong': 37} |
| m2-embed-1.7B | babilong:qa6 | 64k | {'ok': 100} | 61.0 [51.0, 71.0] | 61.0 [51.0, 71.0] | 0.7871 | 5.454 | 7.911 | {'wrong': 39, 'correct': 61} |
| m2-embed-1.7B | babilong:qa6 | 128k | {'ok': 100} | 61.0 [51.0, 70.0] | 61.0 [51.0, 70.0] | 0.5638 | 5.485 | 7.915 | {'correct': 61, 'wrong': 39} |
| m2-embed-1.7B | babilong:qa6 | 256k | {'ok': 100} | 65.0 [55.0, 74.0] | 65.0 [55.0, 74.0] | 0.4608 | 5.545 | 7.915 | {'correct': 65, 'wrong': 35} |
| m2-embed-1.7B | babilong:qa7 | 32k | {'ok': 100} | 0.0 [0.0, 0.0] | 9.0 [4.0, 15.0] | 0.9815 | 5.46 | 7.84 | {'wrong': 28, 'format': 72} |
| m2-embed-1.7B | babilong:qa7 | 64k | {'ok': 100} | 0.0 [0.0, 0.0] | 25.0 [17.0, 34.0] | 0.7637 | 5.523 | 7.882 | {'wrong': 46, 'format': 54} |
| m2-embed-1.7B | babilong:qa7 | 128k | {'ok': 100} | 0.0 [0.0, 0.0] | 29.0 [21.0, 38.0] | 0.5963 | 5.546 | 7.914 | {'wrong': 60, 'format': 40} |
| m2-embed-1.7B | babilong:qa7 | 256k | {'ok': 100} | 0.0 [0.0, 0.0] | 30.0 [21.0, 39.0] | 0.4709 | 5.604 | 7.915 | {'format': 42, 'wrong': 58} |
| m2-embed-1.7B | babilong:qa8 | 32k | {'ok': 100} | 24.0 [16.0, 33.0] | 24.0 [16.0, 33.0] | 0.978 | 5.547 | 7.844 | {'format': 38, 'correct': 24, 'wrong': 38} |
| m2-embed-1.7B | babilong:qa8 | 64k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 0.7809 | 5.617 | 7.912 | {'correct': 30, 'format': 36, 'wrong': 34} |
| m2-embed-1.7B | babilong:qa8 | 128k | {'ok': 100} | 28.0 [19.0, 37.0] | 28.0 [19.0, 37.0] | 0.6013 | 5.651 | 7.915 | {'correct': 28, 'wrong': 48, 'format': 24} |
| m2-embed-1.7B | babilong:qa8 | 256k | {'ok': 100} | 27.0 [19.0, 36.0] | 27.0 [19.0, 36.0] | 0.4689 | 5.692 | 7.917 | {'format': 14, 'wrong': 59, 'correct': 27} |
| m2-embed-1.7B | babilong:qa9 | 32k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 0.9865 | 5.375 | 7.838 | {'correct': 69, 'wrong': 31} |
| m2-embed-1.7B | babilong:qa9 | 64k | {'ok': 100} | 66.0 [57.0, 75.0] | 66.0 [57.0, 75.0] | 0.8175 | 5.452 | 7.911 | {'correct': 66, 'wrong': 34} |
| m2-embed-1.7B | babilong:qa9 | 128k | {'ok': 100} | 73.0 [64.0, 81.0] | 73.0 [64.0, 81.0] | 0.6169 | 5.478 | 7.914 | {'correct': 73, 'wrong': 27} |
| m2-embed-1.7B | babilong:qa9 | 256k | {'ok': 100} | 68.0 [58.0, 77.0] | 68.0 [58.0, 77.0] | 0.5159 | 5.533 | 7.916 | {'correct': 68, 'wrong': 32} |
| m2-embed-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 17.0 [10.0, 25.0] | – | 0.51 | 8.992 | 7.917 | {'correct': 17, 'format': 3, 'wrong': 80} |
| m2-embed-1.7B | infbench:longbook_choice_eng | full | {'ok': 100} | 49.0 [39.0, 59.0] | – | None | 5.924 | 7.914 | {'wrong': 51, 'correct': 49} |
| m2-embed-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.595 | 7.916 | {'correct': 100} |
| m2-embed-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.25 | 7.916 | {'correct': 100} |
| m2-embed-1.7B | synth:S1 | 32k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.741 | 7.834 | {'correct': 100} |
| m2-embed-1.7B | synth:S1 | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.824 | 7.911 | {'correct': 100} |
| m2-embed-1.7B | synth:S1 | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.846 | 7.914 | {'correct': 100} |
| m2-embed-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.913 | 7.916 | {'correct': 100} |
| m2-embed-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 75.0 [66.0, 83.0] | – | 1.0 | 5.74 | 7.835 | {'correct': 75, 'wrong': 25} |
| m2-embed-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 83.0 [75.0, 90.0] | – | 1.0 | 5.832 | 7.911 | {'correct': 83, 'wrong': 17} |
| m2-embed-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 84.0 [77.0, 91.0] | – | 1.0 | 5.851 | 7.914 | {'correct': 84, 'wrong': 16} |
| m2-embed-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 87.0 [80.0, 93.0] | – | 1.0 | 5.911 | 7.915 | {'correct': 87, 'wrong': 13} |
| m2-embed-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 61.0 [52.0, 71.0] | – | 1.0 | 8.473 | 7.842 | {'correct': 61, 'wrong': 39} |
| m2-embed-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 67.0 [58.0, 76.0] | – | 1.0 | 8.726 | 7.913 | {'wrong': 31, 'correct': 67, 'format': 2} |
| m2-embed-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 64.0 [54.0, 73.0] | – | 1.0 | 8.55 | 7.914 | {'wrong': 36, 'correct': 64} |
| m2-embed-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 67.0 [58.0, 76.0] | – | 0.9925 | 9.155 | 7.917 | {'correct': 67, 'wrong': 32, 'format': 1} |
| m2-hybrid-1.7B | babilong:qa1 | 32k | {'ok': 100} | 65.0 [56.0, 74.0] | 65.0 [56.0, 74.0] | 0.982 | 6.052 | 8.969 | {'format': 9, 'correct': 65, 'wrong': 26} |
| m2-hybrid-1.7B | babilong:qa1 | 64k | {'ok': 100} | 57.0 [47.0, 66.0] | 57.0 [47.0, 66.0] | 0.7867 | 6.092 | 9.0 | {'correct': 57, 'wrong': 25, 'format': 18} |
| m2-hybrid-1.7B | babilong:qa1 | 128k | {'ok': 100} | 43.0 [33.0, 53.0] | 43.0 [33.0, 53.0] | 0.6043 | 6.187 | 9.011 | {'format': 34, 'correct': 43, 'wrong': 23} |
| m2-hybrid-1.7B | babilong:qa1 | 256k | {'ok': 100} | 48.0 [38.0, 58.0] | 48.0 [38.0, 58.0] | 0.5546 | 6.19 | 9.027 | {'correct': 48, 'format': 27, 'wrong': 25} |
| m2-hybrid-1.7B | babilong:qa10 | 32k | {'ok': 100} | 57.0 [47.0, 67.0] | 57.0 [47.0, 67.0] | 0.9918 | 5.482 | 9.02 | {'correct': 57, 'wrong': 43} |
| m2-hybrid-1.7B | babilong:qa10 | 64k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 0.8849 | 5.541 | 9.019 | {'correct': 45, 'wrong': 55} |
| m2-hybrid-1.7B | babilong:qa10 | 128k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 0.7381 | 5.611 | 9.023 | {'wrong': 55, 'correct': 45} |
| m2-hybrid-1.7B | babilong:qa10 | 256k | {'ok': 100} | 49.0 [39.0, 59.0] | 49.0 [39.0, 59.0] | 0.6643 | 5.634 | 9.026 | {'correct': 49, 'wrong': 51} |
| m2-hybrid-1.7B | babilong:qa2 | 32k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 0.9674 | 6.114 | 8.961 | {'correct': 30, 'format': 6, 'wrong': 64} |
| m2-hybrid-1.7B | babilong:qa2 | 64k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 0.6411 | 5.913 | 9.005 | {'wrong': 58, 'correct': 31, 'format': 11} |
| m2-hybrid-1.7B | babilong:qa2 | 128k | {'ok': 100} | 14.0 [8.0, 21.0] | 14.0 [8.0, 21.0] | 0.4259 | 6.023 | 9.015 | {'format': 28, 'wrong': 58, 'correct': 14} |
| m2-hybrid-1.7B | babilong:qa2 | 256k | {'ok': 100} | 14.0 [8.0, 21.0] | 14.0 [8.0, 21.0] | 0.2993 | 6.031 | 9.026 | {'correct': 14, 'wrong': 55, 'format': 31} |
| m2-hybrid-1.7B | babilong:qa3 | 32k | {'ok': 100} | 34.0 [25.0, 43.0] | 34.0 [25.0, 43.0] | 0.9708 | 6.172 | 8.96 | {'wrong': 66, 'correct': 34} |
| m2-hybrid-1.7B | babilong:qa3 | 64k | {'ok': 100} | 29.0 [20.0, 38.0] | 29.0 [20.0, 38.0] | 0.669 | 6.132 | 9.011 | {'wrong': 71, 'correct': 29} |
| m2-hybrid-1.7B | babilong:qa3 | 128k | {'ok': 100} | 23.0 [15.0, 31.0] | 23.0 [15.0, 31.0] | 0.4631 | 6.197 | 9.02 | {'wrong': 77, 'correct': 23} |
| m2-hybrid-1.7B | babilong:qa3 | 256k | {'ok': 100} | 24.0 [16.0, 33.0] | 24.0 [16.0, 33.0] | 0.3718 | 6.222 | 9.028 | {'wrong': 76, 'correct': 24} |
| m2-hybrid-1.7B | babilong:qa4 | 32k | {'ok': 100} | 20.0 [13.0, 28.0] | 20.0 [13.0, 28.0] | 1.0 | 5.613 | 8.945 | {'wrong': 76, 'correct': 20, 'format': 4} |
| m2-hybrid-1.7B | babilong:qa4 | 64k | {'ok': 100} | 16.0 [9.0, 24.0] | 16.0 [9.0, 24.0] | 1.0 | 5.619 | 9.002 | {'correct': 16, 'wrong': 80, 'format': 4} |
| m2-hybrid-1.7B | babilong:qa4 | 128k | {'ok': 100} | 18.0 [11.0, 26.0] | 18.0 [11.0, 26.0] | 1.0 | 5.699 | 9.006 | {'wrong': 80, 'correct': 18, 'format': 2} |
| m2-hybrid-1.7B | babilong:qa4 | 256k | {'ok': 100} | 16.0 [9.0, 23.0] | 16.0 [9.0, 23.0] | 0.96 | 5.713 | 9.02 | {'wrong': 82, 'correct': 16, 'format': 2} |
| m2-hybrid-1.7B | babilong:qa5 | 32k | {'ok': 100} | 61.0 [51.0, 70.0] | 61.0 [51.0, 70.0] | 0.985 | 5.663 | 8.955 | {'correct': 61, 'wrong': 39} |
| m2-hybrid-1.7B | babilong:qa5 | 64k | {'ok': 100} | 65.0 [55.0, 74.0] | 65.0 [55.0, 74.0] | 0.7686 | 5.553 | 9.007 | {'correct': 65, 'wrong': 35} |
| m2-hybrid-1.7B | babilong:qa5 | 128k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 0.6406 | 5.637 | 9.014 | {'correct': 70, 'wrong': 30} |
| m2-hybrid-1.7B | babilong:qa5 | 256k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 0.5678 | 5.647 | 9.023 | {'correct': 63, 'wrong': 37} |
| m2-hybrid-1.7B | babilong:qa6 | 32k | {'ok': 100} | 60.0 [50.0, 69.0] | 60.0 [50.0, 69.0] | 0.9898 | 5.421 | 9.021 | {'correct': 60, 'wrong': 40} |
| m2-hybrid-1.7B | babilong:qa6 | 64k | {'ok': 100} | 62.0 [52.0, 72.0] | 62.0 [52.0, 72.0] | 0.7955 | 5.534 | 9.018 | {'correct': 62, 'wrong': 38} |
| m2-hybrid-1.7B | babilong:qa6 | 128k | {'ok': 100} | 56.0 [46.0, 66.0] | 56.0 [46.0, 66.0] | 0.6447 | 5.606 | 9.021 | {'correct': 56, 'wrong': 44} |
| m2-hybrid-1.7B | babilong:qa6 | 256k | {'ok': 100} | 67.0 [58.0, 76.0] | 67.0 [58.0, 76.0] | 0.5634 | 5.632 | 9.025 | {'correct': 67, 'wrong': 33} |
| m2-hybrid-1.7B | babilong:qa7 | 32k | {'ok': 100} | 0.0 [0.0, 0.0] | 11.0 [5.0, 18.0] | 0.9825 | 5.535 | 8.953 | {'wrong': 34, 'format': 66} |
| m2-hybrid-1.7B | babilong:qa7 | 64k | {'ok': 100} | 0.0 [0.0, 0.0] | 20.0 [12.0, 28.0] | 0.7884 | 5.603 | 9.019 | {'wrong': 46, 'format': 54} |
| m2-hybrid-1.7B | babilong:qa7 | 128k | {'ok': 100} | 0.0 [0.0, 0.0] | 34.0 [25.0, 43.0] | 0.6399 | 5.651 | 9.023 | {'wrong': 67, 'format': 33} |
| m2-hybrid-1.7B | babilong:qa7 | 256k | {'ok': 100} | 0.0 [0.0, 0.0] | 24.0 [16.0, 33.0] | 0.5709 | 5.699 | 9.026 | {'wrong': 52, 'format': 48} |
| m2-hybrid-1.7B | babilong:qa8 | 32k | {'ok': 100} | 19.0 [12.0, 27.0] | 19.0 [12.0, 27.0] | 0.9855 | 5.842 | 8.999 | {'format': 40, 'wrong': 41, 'correct': 19} |
| m2-hybrid-1.7B | babilong:qa8 | 64k | {'ok': 100} | 27.0 [18.0, 36.0] | 27.0 [18.0, 36.0] | 0.7927 | 5.692 | 9.018 | {'format': 31, 'correct': 27, 'wrong': 42} |
| m2-hybrid-1.7B | babilong:qa8 | 128k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 0.6382 | 5.747 | 9.024 | {'correct': 30, 'wrong': 49, 'format': 21} |
| m2-hybrid-1.7B | babilong:qa8 | 256k | {'ok': 100} | 29.0 [20.0, 38.0] | 29.0 [20.0, 38.0] | 0.5599 | 5.784 | 9.026 | {'correct': 29, 'wrong': 57, 'format': 14} |
| m2-hybrid-1.7B | babilong:qa9 | 32k | {'ok': 100} | 68.0 [59.0, 77.0] | 68.0 [59.0, 77.0] | 0.9923 | 5.635 | 8.95 | {'correct': 68, 'wrong': 32} |
| m2-hybrid-1.7B | babilong:qa9 | 64k | {'ok': 100} | 68.0 [58.0, 77.0] | 68.0 [58.0, 77.0] | 0.8823 | 5.534 | 9.018 | {'correct': 68, 'wrong': 32} |
| m2-hybrid-1.7B | babilong:qa9 | 128k | {'ok': 100} | 73.0 [64.0, 81.0] | 73.0 [64.0, 81.0] | 0.7403 | 5.593 | 9.022 | {'correct': 73, 'wrong': 27} |
| m2-hybrid-1.7B | babilong:qa9 | 256k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 0.6804 | 5.629 | 9.025 | {'correct': 69, 'wrong': 31} |
| m2-hybrid-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 18.0 [11.0, 26.0] | – | 0.52 | 9.008 | 9.026 | {'correct': 18, 'format': 4, 'wrong': 78} |
| m2-hybrid-1.7B | infbench:longbook_choice_eng | full | {'ok': 100} | 51.0 [41.0, 61.0] | – | None | 6.115 | 9.023 | {'wrong': 48, 'correct': 51, 'format': 1} |
| m2-hybrid-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.882 | 9.027 | {'correct': 100} |
| m2-hybrid-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.333 | 9.025 | {'correct': 100} |
| m2-hybrid-1.7B | synth:S1 | 32k | {'ok': 100} | 99.0 [97.0, 100.0] | – | 1.0 | 5.772 | 8.946 | {'correct': 99, 'format': 1} |
| m2-hybrid-1.7B | synth:S1 | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.888 | 8.985 | {'correct': 100} |
| m2-hybrid-1.7B | synth:S1 | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.951 | 9.008 | {'correct': 100} |
| m2-hybrid-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.025 | 9.025 | {'correct': 100} |
| m2-hybrid-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 75.0 [66.0, 83.0] | – | 1.0 | 5.784 | 8.945 | {'correct': 75, 'wrong': 25} |
| m2-hybrid-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 86.0 [79.0, 92.0] | – | 1.0 | 5.889 | 8.986 | {'correct': 86, 'wrong': 14} |
| m2-hybrid-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 76.0 [68.0, 84.0] | – | 1.0 | 5.951 | 9.008 | {'correct': 76, 'wrong': 24} |
| m2-hybrid-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 84.0 [77.0, 91.0] | – | 1.0 | 6.003 | 9.025 | {'correct': 84, 'wrong': 16} |
| m2-hybrid-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 52.0 [42.0, 62.0] | – | 1.0 | 8.226 | 8.962 | {'correct': 52, 'wrong': 47, 'format': 1} |
| m2-hybrid-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 66.0 [57.0, 75.0] | – | 1.0 | 8.444 | 8.991 | {'correct': 66, 'wrong': 34} |
| m2-hybrid-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 64.0 [55.0, 73.0] | – | 1.0 | 9.298 | 9.013 | {'correct': 64, 'format': 3, 'wrong': 33} |
| m2-hybrid-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 71.0 [62.0, 80.0] | – | 1.0 | 9.289 | 9.026 | {'correct': 71, 'wrong': 28, 'format': 1} |
| m2-hybrid-rerank-1.7B | babilong:qa1 | 32k | {'ok': 100} | 65.0 [56.0, 74.0] | 65.0 [56.0, 74.0] | 0.982 | None | 6.569 | {'format': 9, 'correct': 65, 'wrong': 26} |
| m2-hybrid-rerank-1.7B | babilong:qa1 | 64k | {'ok': 100} | 57.0 [47.0, 66.0] | 57.0 [47.0, 66.0] | 0.7867 | None | 6.577 | {'correct': 57, 'wrong': 25, 'format': 18} |
| m2-hybrid-rerank-1.7B | babilong:qa1 | 128k | {'ok': 100} | 43.0 [33.0, 53.0] | 43.0 [33.0, 53.0] | 0.6043 | None | 6.577 | {'format': 34, 'correct': 43, 'wrong': 23} |
| m2-hybrid-rerank-1.7B | babilong:qa1 | 256k | {'ok': 100} | 48.0 [38.0, 58.0] | 48.0 [38.0, 58.0] | 0.5546 | None | 6.56 | {'correct': 48, 'format': 27, 'wrong': 25} |
| m2-hybrid-rerank-1.7B | babilong:qa10 | 32k | {'ok': 100} | 57.0 [47.0, 67.0] | 57.0 [47.0, 67.0] | 0.9918 | None | 6.582 | {'correct': 57, 'wrong': 43} |
| m2-hybrid-rerank-1.7B | babilong:qa10 | 64k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 0.8849 | None | 6.578 | {'correct': 45, 'wrong': 55} |
| m2-hybrid-rerank-1.7B | babilong:qa10 | 128k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 0.7381 | None | 6.586 | {'wrong': 55, 'correct': 45} |
| m2-hybrid-rerank-1.7B | babilong:qa10 | 256k | {'ok': 100} | 49.0 [39.0, 59.0] | 49.0 [39.0, 59.0] | 0.6643 | None | 6.586 | {'correct': 49, 'wrong': 51} |
| m2-hybrid-rerank-1.7B | babilong:qa2 | 32k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 0.9674 | None | 6.577 | {'correct': 30, 'format': 6, 'wrong': 64} |
| m2-hybrid-rerank-1.7B | babilong:qa2 | 64k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 0.6411 | None | 6.586 | {'wrong': 58, 'correct': 31, 'format': 11} |
| m2-hybrid-rerank-1.7B | babilong:qa2 | 128k | {'ok': 100} | 14.0 [8.0, 21.0] | 14.0 [8.0, 21.0] | 0.4259 | None | 6.586 | {'format': 28, 'wrong': 58, 'correct': 14} |
| m2-hybrid-rerank-1.7B | babilong:qa2 | 256k | {'ok': 100} | 14.0 [8.0, 21.0] | 14.0 [8.0, 21.0] | 0.2993 | None | 6.582 | {'correct': 14, 'wrong': 55, 'format': 31} |
| m2-hybrid-rerank-1.7B | babilong:qa3 | 32k | {'ok': 100} | 34.0 [25.0, 43.0] | 34.0 [25.0, 43.0] | 0.9708 | None | 6.59 | {'wrong': 66, 'correct': 34} |
| m2-hybrid-rerank-1.7B | babilong:qa3 | 64k | {'ok': 100} | 29.0 [20.0, 38.0] | 29.0 [20.0, 38.0] | 0.669 | None | 6.578 | {'wrong': 71, 'correct': 29} |
| m2-hybrid-rerank-1.7B | babilong:qa3 | 128k | {'ok': 100} | 23.0 [15.0, 31.0] | 23.0 [15.0, 31.0] | 0.4631 | None | 6.595 | {'wrong': 77, 'correct': 23} |
| m2-hybrid-rerank-1.7B | babilong:qa3 | 256k | {'ok': 100} | 24.0 [16.0, 33.0] | 24.0 [16.0, 33.0] | 0.3718 | None | 6.578 | {'wrong': 76, 'correct': 24} |
| m2-hybrid-rerank-1.7B | babilong:qa4 | 32k | {'ok': 100} | 20.0 [13.0, 28.0] | 20.0 [13.0, 28.0] | 1.0 | None | 6.578 | {'wrong': 76, 'correct': 20, 'format': 4} |
| m2-hybrid-rerank-1.7B | babilong:qa4 | 64k | {'ok': 100} | 16.0 [9.0, 24.0] | 16.0 [9.0, 24.0] | 1.0 | None | 6.591 | {'correct': 16, 'wrong': 80, 'format': 4} |
| m2-hybrid-rerank-1.7B | babilong:qa4 | 128k | {'ok': 100} | 18.0 [11.0, 26.0] | 18.0 [11.0, 26.0] | 1.0 | None | 6.578 | {'wrong': 80, 'correct': 18, 'format': 2} |
| m2-hybrid-rerank-1.7B | babilong:qa4 | 256k | {'ok': 100} | 16.0 [9.0, 23.0] | 16.0 [9.0, 23.0] | 0.96 | None | 6.561 | {'wrong': 82, 'correct': 16, 'format': 2} |
| m2-hybrid-rerank-1.7B | babilong:qa5 | 32k | {'ok': 100} | 61.0 [51.0, 70.0] | 61.0 [51.0, 70.0] | 0.985 | None | 6.573 | {'correct': 61, 'wrong': 39} |
| m2-hybrid-rerank-1.7B | babilong:qa5 | 64k | {'ok': 100} | 65.0 [55.0, 74.0] | 65.0 [55.0, 74.0] | 0.7686 | None | 6.578 | {'correct': 65, 'wrong': 35} |
| m2-hybrid-rerank-1.7B | babilong:qa5 | 128k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 0.6406 | None | 6.569 | {'correct': 70, 'wrong': 30} |
| m2-hybrid-rerank-1.7B | babilong:qa5 | 256k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 0.5678 | None | 6.59 | {'correct': 63, 'wrong': 37} |
| m2-hybrid-rerank-1.7B | babilong:qa6 | 32k | {'ok': 100} | 60.0 [50.0, 69.0] | 60.0 [50.0, 69.0] | 0.9898 | None | 6.577 | {'correct': 60, 'wrong': 40} |
| m2-hybrid-rerank-1.7B | babilong:qa6 | 64k | {'ok': 100} | 62.0 [52.0, 72.0] | 62.0 [52.0, 72.0] | 0.7955 | None | 6.586 | {'correct': 62, 'wrong': 38} |
| m2-hybrid-rerank-1.7B | babilong:qa6 | 128k | {'ok': 100} | 56.0 [46.0, 66.0] | 56.0 [46.0, 66.0] | 0.6447 | None | 6.599 | {'correct': 56, 'wrong': 44} |
| m2-hybrid-rerank-1.7B | babilong:qa6 | 256k | {'ok': 100} | 67.0 [58.0, 76.0] | 67.0 [58.0, 76.0] | 0.5634 | None | 6.578 | {'correct': 67, 'wrong': 33} |
| m2-hybrid-rerank-1.7B | babilong:qa7 | 32k | {'ok': 100} | 0.0 [0.0, 0.0] | 11.0 [5.0, 18.0] | 0.9825 | None | 6.603 | {'wrong': 34, 'format': 66} |
| m2-hybrid-rerank-1.7B | babilong:qa7 | 64k | {'ok': 100} | 0.0 [0.0, 0.0] | 20.0 [12.0, 28.0] | 0.7884 | None | 6.586 | {'wrong': 46, 'format': 54} |
| m2-hybrid-rerank-1.7B | babilong:qa7 | 128k | {'ok': 100} | 0.0 [0.0, 0.0] | 34.0 [25.0, 43.0] | 0.6399 | None | 6.599 | {'wrong': 67, 'format': 33} |
| m2-hybrid-rerank-1.7B | babilong:qa7 | 256k | {'ok': 100} | 0.0 [0.0, 0.0] | 24.0 [16.0, 33.0] | 0.5709 | None | 6.595 | {'wrong': 52, 'format': 48} |
| m2-hybrid-rerank-1.7B | babilong:qa8 | 32k | {'ok': 100} | 19.0 [12.0, 27.0] | 19.0 [12.0, 27.0] | 0.9855 | None | 6.59 | {'format': 40, 'wrong': 41, 'correct': 19} |
| m2-hybrid-rerank-1.7B | babilong:qa8 | 64k | {'ok': 100} | 27.0 [18.0, 36.0] | 27.0 [18.0, 36.0] | 0.7927 | None | 6.569 | {'format': 31, 'correct': 27, 'wrong': 42} |
| m2-hybrid-rerank-1.7B | babilong:qa8 | 128k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 0.6382 | None | 6.582 | {'correct': 30, 'wrong': 49, 'format': 21} |
| m2-hybrid-rerank-1.7B | babilong:qa8 | 256k | {'ok': 100} | 29.0 [20.0, 38.0] | 29.0 [20.0, 38.0] | 0.5599 | None | 6.59 | {'correct': 29, 'wrong': 57, 'format': 14} |
| m2-hybrid-rerank-1.7B | babilong:qa9 | 32k | {'ok': 100} | 68.0 [59.0, 77.0] | 68.0 [59.0, 77.0] | 0.9923 | None | 6.586 | {'correct': 68, 'wrong': 32} |
| m2-hybrid-rerank-1.7B | babilong:qa9 | 64k | {'ok': 100} | 68.0 [58.0, 77.0] | 68.0 [58.0, 77.0] | 0.8823 | None | 6.595 | {'correct': 68, 'wrong': 32} |
| m2-hybrid-rerank-1.7B | babilong:qa9 | 128k | {'ok': 100} | 73.0 [64.0, 81.0] | 73.0 [64.0, 81.0] | 0.7403 | None | 6.59 | {'correct': 73, 'wrong': 27} |
| m2-hybrid-rerank-1.7B | babilong:qa9 | 256k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 0.6804 | None | 6.582 | {'correct': 69, 'wrong': 31} |
| m2-hybrid-rerank-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 18.0 [11.0, 26.0] | – | 0.52 | None | 6.515 | {'correct': 18, 'format': 4, 'wrong': 78} |
| m2-hybrid-rerank-1.7B | infbench:longbook_choice_eng | full | {'ok': 100} | 51.0 [41.0, 61.0] | – | None | None | 6.81 | {'wrong': 48, 'correct': 51, 'format': 1} |
| m2-hybrid-rerank-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | None | 6.358 | {'correct': 100} |
| m2-hybrid-rerank-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | None | 6.346 | {'correct': 100} |
| m2-hybrid-rerank-1.7B | synth:S1 | 32k | {'ok': 100} | 99.0 [97.0, 100.0] | – | 1.0 | None | 6.655 | {'correct': 99, 'format': 1} |
| m2-hybrid-rerank-1.7B | synth:S1 | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | None | 6.663 | {'correct': 100} |
| m2-hybrid-rerank-1.7B | synth:S1 | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | None | 6.655 | {'correct': 100} |
| m2-hybrid-rerank-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | None | 6.676 | {'correct': 100} |
| m2-hybrid-rerank-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 75.0 [66.0, 83.0] | – | 1.0 | None | 6.651 | {'correct': 75, 'wrong': 25} |
| m2-hybrid-rerank-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 86.0 [79.0, 92.0] | – | 1.0 | None | 6.642 | {'correct': 86, 'wrong': 14} |
| m2-hybrid-rerank-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 76.0 [68.0, 84.0] | – | 1.0 | None | 6.646 | {'correct': 76, 'wrong': 24} |
| m2-hybrid-rerank-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 84.0 [77.0, 91.0] | – | 1.0 | None | 6.663 | {'correct': 84, 'wrong': 16} |
| m2-hybrid-rerank-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 52.0 [42.0, 62.0] | – | 1.0 | None | 6.68 | {'correct': 52, 'wrong': 47, 'format': 1} |
| m2-hybrid-rerank-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 66.0 [57.0, 75.0] | – | 1.0 | None | 6.681 | {'correct': 66, 'wrong': 34} |
| m2-hybrid-rerank-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 64.0 [55.0, 73.0] | – | 1.0 | None | 6.664 | {'correct': 64, 'format': 3, 'wrong': 33} |
| m2-hybrid-rerank-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 71.0 [62.0, 80.0] | – | 1.0 | None | 6.668 | {'correct': 71, 'wrong': 28, 'format': 1} |
| m3-oracle-1.7B | babilong:qa1 | 32k | {'ok': 100} | 66.0 [56.0, 75.0] | 66.0 [56.0, 75.0] | 1.0 | 5.835 | 6.748 | {'format': 7, 'correct': 66, 'wrong': 27} |
| m3-oracle-1.7B | babilong:qa1 | 64k | {'ok': 100} | 66.0 [57.0, 75.0] | 66.0 [57.0, 75.0] | 1.0 | 5.855 | 6.725 | {'correct': 66, 'wrong': 27, 'format': 7} |
| m3-oracle-1.7B | babilong:qa1 | 128k | {'ok': 100} | 59.0 [49.0, 69.0] | 59.0 [49.0, 69.0] | 1.0 | 6.386 | 6.724 | {'correct': 59, 'format': 13, 'wrong': 28} |
| m3-oracle-1.7B | babilong:qa1 | 256k | {'ok': 100} | 62.0 [52.0, 71.0] | 62.0 [52.0, 71.0] | 1.0 | 5.856 | 6.725 | {'correct': 62, 'format': 14, 'wrong': 24} |
| m3-oracle-1.7B | babilong:qa10 | 32k | {'ok': 100} | 53.0 [43.0, 63.0] | 53.0 [43.0, 63.0] | 1.0 | 5.319 | 6.801 | {'correct': 53, 'wrong': 47} |
| m3-oracle-1.7B | babilong:qa10 | 64k | {'ok': 100} | 45.0 [35.0, 55.0] | 45.0 [35.0, 55.0] | 1.0 | 5.354 | 6.727 | {'wrong': 55, 'correct': 45} |
| m3-oracle-1.7B | babilong:qa10 | 128k | {'ok': 100} | 53.0 [43.0, 63.0] | 53.0 [43.0, 63.0] | 1.0 | 5.332 | 6.728 | {'correct': 53, 'wrong': 47} |
| m3-oracle-1.7B | babilong:qa10 | 256k | {'ok': 100} | 48.0 [38.0, 58.0] | 48.0 [38.0, 58.0] | 1.0 | 5.311 | 6.728 | {'wrong': 52, 'correct': 48} |
| m3-oracle-1.7B | babilong:qa2 | 32k | {'ok': 100} | 28.0 [19.0, 37.0] | 28.0 [19.0, 37.0] | 1.0 | 5.665 | 6.726 | {'correct': 28, 'wrong': 67, 'format': 5} |
| m3-oracle-1.7B | babilong:qa2 | 64k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 1.0 | 5.685 | 6.729 | {'wrong': 64, 'correct': 31, 'format': 5} |
| m3-oracle-1.7B | babilong:qa2 | 128k | {'ok': 100} | 39.0 [30.0, 49.0] | 39.0 [30.0, 49.0] | 1.0 | 5.954 | 6.734 | {'format': 6, 'wrong': 55, 'correct': 39} |
| m3-oracle-1.7B | babilong:qa2 | 256k | {'ok': 100} | 33.0 [24.0, 42.0] | 33.0 [24.0, 42.0] | 1.0 | 5.694 | 6.737 | {'wrong': 63, 'correct': 33, 'format': 4} |
| m3-oracle-1.7B | babilong:qa3 | 32k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 1.0 | 5.857 | 6.734 | {'wrong': 69, 'correct': 31} |
| m3-oracle-1.7B | babilong:qa3 | 64k | {'ok': 100} | 35.0 [26.0, 45.0] | 35.0 [26.0, 45.0] | 1.0 | 5.906 | 6.746 | {'wrong': 65, 'correct': 35} |
| m3-oracle-1.7B | babilong:qa3 | 128k | {'ok': 100} | 36.0 [27.0, 45.0] | 36.0 [27.0, 45.0] | 1.0 | 6.42 | 6.808 | {'wrong': 64, 'correct': 36} |
| m3-oracle-1.7B | babilong:qa3 | 256k | {'ok': 100} | 34.0 [25.0, 44.0] | 34.0 [25.0, 44.0] | 1.0 | 5.928 | 6.807 | {'wrong': 66, 'correct': 34} |
| m3-oracle-1.7B | babilong:qa4 | 32k | {'ok': 100} | 23.0 [15.0, 31.0] | 23.0 [15.0, 31.0] | 1.0 | 5.481 | 6.718 | {'wrong': 74, 'correct': 23, 'format': 3} |
| m3-oracle-1.7B | babilong:qa4 | 64k | {'ok': 100} | 32.0 [23.0, 41.0] | 32.0 [23.0, 41.0] | 1.0 | 5.48 | 6.719 | {'correct': 32, 'wrong': 62, 'format': 6} |
| m3-oracle-1.7B | babilong:qa4 | 128k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 1.0 | 5.774 | 6.719 | {'wrong': 65, 'correct': 30, 'format': 5} |
| m3-oracle-1.7B | babilong:qa4 | 256k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 1.0 | 5.538 | 6.719 | {'wrong': 66, 'correct': 31, 'format': 3} |
| m3-oracle-1.7B | babilong:qa5 | 32k | {'ok': 100} | 62.0 [52.0, 71.0] | 62.0 [52.0, 71.0] | 1.0 | 5.295 | 6.726 | {'correct': 62, 'wrong': 38} |
| m3-oracle-1.7B | babilong:qa5 | 64k | {'ok': 100} | 63.0 [53.0, 72.0] | 63.0 [53.0, 72.0] | 1.0 | 5.319 | 6.733 | {'correct': 63, 'wrong': 37} |
| m3-oracle-1.7B | babilong:qa5 | 128k | {'ok': 100} | 61.0 [51.0, 70.0] | 61.0 [51.0, 70.0] | 1.0 | 5.402 | 6.739 | {'correct': 61, 'wrong': 39} |
| m3-oracle-1.7B | babilong:qa5 | 256k | {'ok': 100} | 64.0 [54.0, 73.0] | 64.0 [54.0, 73.0] | 1.0 | 5.331 | 6.738 | {'correct': 64, 'wrong': 36} |
| m3-oracle-1.7B | babilong:qa6 | 32k | {'ok': 100} | 61.0 [51.0, 70.0] | 61.0 [51.0, 70.0] | 1.0 | 5.325 | 6.802 | {'correct': 61, 'wrong': 39} |
| m3-oracle-1.7B | babilong:qa6 | 64k | {'ok': 100} | 60.0 [50.0, 70.0] | 60.0 [50.0, 70.0] | 1.0 | 5.352 | 6.724 | {'wrong': 40, 'correct': 60} |
| m3-oracle-1.7B | babilong:qa6 | 128k | {'ok': 100} | 59.0 [49.0, 69.0] | 59.0 [49.0, 69.0] | 1.0 | 5.307 | 6.725 | {'correct': 59, 'wrong': 41} |
| m3-oracle-1.7B | babilong:qa6 | 256k | {'ok': 100} | 60.0 [50.0, 70.0] | 60.0 [50.0, 70.0] | 1.0 | 5.312 | 6.725 | {'correct': 60, 'wrong': 40} |
| m3-oracle-1.7B | babilong:qa7 | 32k | {'ok': 100} | 0.0 [0.0, 0.0] | 11.0 [5.0, 17.0] | 1.0 | 5.435 | 6.724 | {'wrong': 30, 'format': 70} |
| m3-oracle-1.7B | babilong:qa7 | 64k | {'ok': 100} | 0.0 [0.0, 0.0] | 14.0 [8.0, 21.0] | 1.0 | 5.427 | 6.726 | {'wrong': 32, 'format': 68} |
| m3-oracle-1.7B | babilong:qa7 | 128k | {'ok': 100} | 0.0 [0.0, 0.0] | 16.0 [9.0, 23.0] | 1.0 | 5.391 | 6.73 | {'wrong': 32, 'format': 68} |
| m3-oracle-1.7B | babilong:qa7 | 256k | {'ok': 100} | 0.0 [0.0, 0.0] | 14.0 [8.0, 21.0] | 1.0 | 5.397 | 6.729 | {'format': 72, 'wrong': 28} |
| m3-oracle-1.7B | babilong:qa8 | 32k | {'ok': 100} | 22.0 [14.0, 30.0] | 22.0 [14.0, 30.0] | 1.0 | 5.595 | 6.779 | {'wrong': 35, 'correct': 22, 'format': 43} |
| m3-oracle-1.7B | babilong:qa8 | 64k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 1.0 | 5.622 | 6.735 | {'format': 33, 'correct': 30, 'wrong': 37} |
| m3-oracle-1.7B | babilong:qa8 | 128k | {'ok': 100} | 18.0 [11.0, 26.0] | 18.0 [11.0, 26.0] | 1.0 | 5.475 | 6.731 | {'wrong': 41, 'correct': 18, 'format': 41} |
| m3-oracle-1.7B | babilong:qa8 | 256k | {'ok': 100} | 26.0 [18.0, 35.0] | 26.0 [18.0, 35.0] | 1.0 | 5.483 | 6.739 | {'format': 26, 'correct': 26, 'wrong': 48} |
| m3-oracle-1.7B | babilong:qa9 | 32k | {'ok': 100} | 68.0 [58.0, 77.0] | 68.0 [58.0, 77.0] | 1.0 | 5.341 | 6.722 | {'correct': 68, 'wrong': 32} |
| m3-oracle-1.7B | babilong:qa9 | 64k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 1.0 | 5.344 | 6.724 | {'correct': 70, 'wrong': 30} |
| m3-oracle-1.7B | babilong:qa9 | 128k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 1.0 | 5.307 | 6.724 | {'correct': 69, 'wrong': 31} |
| m3-oracle-1.7B | babilong:qa9 | 256k | {'ok': 100} | 67.0 [58.0, 76.0] | 67.0 [58.0, 76.0] | 1.0 | 5.307 | 6.724 | {'correct': 67, 'wrong': 33} |
| m3-oracle-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 21.0 [13.0, 29.0] | – | 1.0 | 9.928 | 6.814 | {'correct': 21, 'wrong': 79} |
| m3-oracle-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 6.287 | 6.79 | {'correct': 100} |
| m3-oracle-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.98 | 6.712 | {'correct': 100} |
| m3-oracle-1.7B | synth:S1 | 32k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.617 | 6.807 | {'correct': 100} |
| m3-oracle-1.7B | synth:S1 | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.642 | 6.806 | {'correct': 100} |
| m3-oracle-1.7B | synth:S1 | 128k | {'ok': 100} | 99.0 [97.0, 100.0] | – | 1.0 | 5.635 | 6.807 | {'correct': 99, 'format': 1} |
| m3-oracle-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 5.637 | 6.807 | {'correct': 100} |
| m3-oracle-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 84.0 [77.0, 91.0] | – | 1.0 | 5.625 | 6.807 | {'correct': 84, 'wrong': 16} |
| m3-oracle-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 94.0 [89.0, 98.0] | – | 1.0 | 5.633 | 6.805 | {'correct': 94, 'wrong': 6} |
| m3-oracle-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 94.0 [89.0, 98.0] | – | 1.0 | 5.649 | 6.805 | {'correct': 94, 'wrong': 6} |
| m3-oracle-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 98.0 [95.0, 100.0] | – | 1.0 | 5.634 | 6.806 | {'correct': 98, 'wrong': 2} |
| m3-oracle-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 55.0 [45.0, 65.0] | – | 1.0 | 8.025 | 6.814 | {'correct': 55, 'wrong': 44, 'format': 1} |
| m3-oracle-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 53.0 [43.0, 63.0] | – | 1.0 | 7.859 | 6.814 | {'wrong': 47, 'correct': 53} |
| m3-oracle-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 64.0 [55.0, 73.0] | – | 1.0 | 8.392 | 6.814 | {'wrong': 35, 'correct': 64, 'format': 1} |
| m3-oracle-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 62.0 [53.0, 72.0] | – | 1.0 | 7.735 | 6.814 | {'correct': 62, 'wrong': 38} |
| m3-oracle-tight-1.7B | babilong:qa1 | 8k | {'ok': 100} | 80.0 [72.0, 87.0] | 80.0 [72.0, 87.0] | 1.0 | 0.747 | 3.75 | {'correct': 80, 'wrong': 19, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa1 | 16k | {'ok': 100} | 81.0 [73.0, 88.0] | 81.0 [73.0, 88.0] | 1.0 | 0.769 | 3.773 | {'correct': 81, 'wrong': 18, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa1 | 32k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.783 | 3.773 | {'correct': 82, 'wrong': 18} |
| m3-oracle-tight-1.7B | babilong:qa1 | 64k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.807 | 3.774 | {'correct': 82, 'wrong': 18} |
| m3-oracle-tight-1.7B | babilong:qa1 | 128k | {'ok': 100} | 79.0 [71.0, 87.0] | 79.0 [71.0, 87.0] | 1.0 | 0.888 | 3.775 | {'correct': 79, 'wrong': 20, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa1 | 256k | {'ok': 100} | 81.0 [73.0, 89.0] | 81.0 [73.0, 89.0] | 1.0 | 0.88 | 3.772 | {'correct': 81, 'wrong': 19} |
| m3-oracle-tight-1.7B | babilong:qa10 | 8k | {'ok': 100} | 61.0 [51.0, 71.0] | 61.0 [51.0, 71.0] | 1.0 | 0.247 | 3.753 | {'correct': 61, 'wrong': 39} |
| m3-oracle-tight-1.7B | babilong:qa10 | 16k | {'ok': 100} | 65.0 [56.0, 74.0] | 65.0 [56.0, 74.0] | 1.0 | 0.267 | 3.771 | {'correct': 65, 'wrong': 35} |
| m3-oracle-tight-1.7B | babilong:qa10 | 32k | {'ok': 100} | 63.0 [54.0, 73.0] | 63.0 [54.0, 73.0] | 1.0 | 0.279 | 3.773 | {'correct': 63, 'wrong': 37} |
| m3-oracle-tight-1.7B | babilong:qa10 | 64k | {'ok': 100} | 61.0 [52.0, 70.0] | 61.0 [52.0, 70.0] | 1.0 | 0.296 | 3.777 | {'correct': 61, 'wrong': 39} |
| m3-oracle-tight-1.7B | babilong:qa10 | 128k | {'ok': 100} | 65.0 [56.0, 74.0] | 65.0 [56.0, 74.0] | 1.0 | 0.297 | 3.775 | {'correct': 65, 'wrong': 35} |
| m3-oracle-tight-1.7B | babilong:qa10 | 256k | {'ok': 100} | 66.0 [57.0, 75.0] | 66.0 [57.0, 75.0] | 1.0 | 0.297 | 3.837 | {'correct': 66, 'wrong': 34} |
| m3-oracle-tight-1.7B | babilong:qa2 | 8k | {'ok': 100} | 52.0 [42.0, 62.0] | 52.0 [42.0, 62.0] | 1.0 | 0.813 | 4.007 | {'correct': 52, 'wrong': 45, 'format': 3} |
| m3-oracle-tight-1.7B | babilong:qa2 | 16k | {'ok': 100} | 47.0 [37.0, 57.0] | 47.0 [37.0, 57.0] | 1.0 | 0.913 | 4.313 | {'correct': 47, 'wrong': 50, 'format': 3} |
| m3-oracle-tight-1.7B | babilong:qa2 | 32k | {'ok': 100} | 44.0 [34.0, 54.0] | 44.0 [34.0, 54.0] | 1.0 | 0.983 | 4.537 | {'correct': 44, 'wrong': 53, 'format': 3} |
| m3-oracle-tight-1.7B | babilong:qa2 | 64k | {'ok': 100} | 48.0 [38.0, 58.0] | 48.0 [38.0, 58.0] | 1.0 | 1.044 | 4.765 | {'correct': 48, 'wrong': 51, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa2 | 128k | {'ok': 100} | 46.0 [36.0, 56.0] | 46.0 [36.0, 56.0] | 1.0 | 1.102 | 4.771 | {'correct': 46, 'wrong': 53, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa2 | 256k | {'ok': 100} | 44.0 [34.0, 54.0] | 44.0 [34.0, 54.0] | 1.0 | 1.188 | 4.839 | {'correct': 44, 'wrong': 52, 'format': 4} |
| m3-oracle-tight-1.7B | babilong:qa3 | 8k | {'ok': 100} | 30.0 [21.0, 39.0] | 30.0 [21.0, 39.0] | 1.0 | 1.25 | 4.26 | {'wrong': 70, 'correct': 30} |
| m3-oracle-tight-1.7B | babilong:qa3 | 16k | {'ok': 100} | 35.0 [26.0, 44.0] | 35.0 [26.0, 44.0] | 1.0 | 1.777 | 5.085 | {'wrong': 65, 'correct': 35} |
| m3-oracle-tight-1.7B | babilong:qa3 | 32k | {'ok': 100} | 33.0 [24.0, 42.0] | 33.0 [24.0, 42.0] | 1.0 | 2.469 | 6.558 | {'wrong': 67, 'correct': 33} |
| m3-oracle-tight-1.7B | babilong:qa3 | 64k | {'ok': 100} | 35.0 [26.0, 44.0] | 35.0 [26.0, 44.0] | 1.0 | 3.034 | 6.808 | {'correct': 35, 'wrong': 65} |
| m3-oracle-tight-1.7B | babilong:qa3 | 128k | {'ok': 100} | 34.0 [25.0, 43.0] | 34.0 [25.0, 43.0] | 1.0 | 3.278 | 6.808 | {'wrong': 66, 'correct': 34} |
| m3-oracle-tight-1.7B | babilong:qa3 | 256k | {'ok': 100} | 31.0 [22.0, 40.0] | 31.0 [22.0, 40.0] | 1.0 | 3.415 | 6.808 | {'wrong': 69, 'correct': 31} |
| m3-oracle-tight-1.7B | babilong:qa4 | 8k | {'ok': 100} | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] | 1.0 | 0.196 | 3.375 | {'correct': 55, 'wrong': 45} |
| m3-oracle-tight-1.7B | babilong:qa4 | 16k | {'ok': 100} | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] | 1.0 | 0.201 | 3.375 | {'correct': 55, 'wrong': 45} |
| m3-oracle-tight-1.7B | babilong:qa4 | 32k | {'ok': 100} | 54.0 [44.0, 64.0] | 54.0 [44.0, 64.0] | 1.0 | 0.218 | 3.371 | {'correct': 54, 'wrong': 45, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa4 | 64k | {'ok': 100} | 53.0 [43.0, 63.0] | 53.0 [43.0, 63.0] | 1.0 | 0.226 | 3.374 | {'wrong': 46, 'correct': 53, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa4 | 128k | {'ok': 100} | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] | 1.0 | 0.21 | 3.373 | {'correct': 55, 'wrong': 45} |
| m3-oracle-tight-1.7B | babilong:qa4 | 256k | {'ok': 100} | 54.0 [44.0, 64.0] | 54.0 [44.0, 64.0] | 1.0 | 0.212 | 3.37 | {'correct': 54, 'wrong': 45, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa5 | 8k | {'ok': 100} | 75.0 [66.0, 83.0] | 75.0 [66.0, 83.0] | 1.0 | 0.449 | 4.128 | {'correct': 75, 'wrong': 25} |
| m3-oracle-tight-1.7B | babilong:qa5 | 16k | {'ok': 100} | 68.0 [59.0, 77.0] | 68.0 [59.0, 77.0] | 1.0 | 0.586 | 4.596 | {'correct': 68, 'wrong': 32} |
| m3-oracle-tight-1.7B | babilong:qa5 | 32k | {'ok': 100} | 67.0 [58.0, 76.0] | 67.0 [58.0, 76.0] | 1.0 | 0.71 | 5.228 | {'correct': 67, 'wrong': 33} |
| m3-oracle-tight-1.7B | babilong:qa5 | 64k | {'ok': 100} | 70.0 [61.0, 79.0] | 70.0 [61.0, 79.0] | 1.0 | 0.787 | 5.515 | {'correct': 70, 'wrong': 30} |
| m3-oracle-tight-1.7B | babilong:qa5 | 128k | {'ok': 100} | 67.0 [58.0, 76.0] | 67.0 [58.0, 76.0] | 1.0 | 0.841 | 5.578 | {'correct': 67, 'wrong': 33} |
| m3-oracle-tight-1.7B | babilong:qa5 | 256k | {'ok': 100} | 69.0 [60.0, 78.0] | 69.0 [60.0, 78.0] | 1.0 | 0.875 | 5.764 | {'correct': 69, 'wrong': 31} |
| m3-oracle-tight-1.7B | babilong:qa6 | 8k | {'ok': 100} | 87.0 [80.0, 93.0] | 87.0 [80.0, 93.0] | 1.0 | 0.245 | 3.756 | {'correct': 87, 'wrong': 13} |
| m3-oracle-tight-1.7B | babilong:qa6 | 16k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.26 | 3.757 | {'correct': 82, 'wrong': 18} |
| m3-oracle-tight-1.7B | babilong:qa6 | 32k | {'ok': 100} | 86.0 [79.0, 92.0] | 86.0 [79.0, 92.0] | 1.0 | 0.275 | 3.769 | {'correct': 86, 'wrong': 14} |
| m3-oracle-tight-1.7B | babilong:qa6 | 64k | {'ok': 100} | 86.0 [79.0, 92.0] | 86.0 [79.0, 92.0] | 1.0 | 0.286 | 3.766 | {'correct': 86, 'wrong': 14} |
| m3-oracle-tight-1.7B | babilong:qa6 | 128k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.291 | 3.763 | {'correct': 82, 'wrong': 18} |
| m3-oracle-tight-1.7B | babilong:qa6 | 256k | {'ok': 100} | 84.0 [77.0, 91.0] | 84.0 [77.0, 91.0] | 1.0 | 0.285 | 3.765 | {'correct': 84, 'wrong': 16} |
| m3-oracle-tight-1.7B | babilong:qa7 | 8k | {'ok': 100} | 4.0 [1.0, 8.0] | 41.0 [31.0, 51.0] | 1.0 | 0.304 | 4.026 | {'wrong': 77, 'format': 19, 'correct': 4} |
| m3-oracle-tight-1.7B | babilong:qa7 | 16k | {'ok': 100} | 5.0 [1.0, 10.0] | 42.0 [32.0, 52.0] | 1.0 | 0.338 | 4.412 | {'wrong': 81, 'format': 14, 'correct': 5} |
| m3-oracle-tight-1.7B | babilong:qa7 | 32k | {'ok': 100} | 6.0 [2.0, 11.0] | 40.0 [31.0, 50.0] | 1.0 | 0.356 | 4.537 | {'wrong': 77, 'format': 17, 'correct': 6} |
| m3-oracle-tight-1.7B | babilong:qa7 | 64k | {'ok': 100} | 5.0 [1.0, 10.0] | 39.0 [30.0, 49.0] | 1.0 | 0.378 | 4.549 | {'wrong': 80, 'format': 15, 'correct': 5} |
| m3-oracle-tight-1.7B | babilong:qa7 | 128k | {'ok': 100} | 7.0 [2.0, 12.0] | 44.0 [34.0, 54.0] | 1.0 | 0.39 | 4.599 | {'wrong': 82, 'format': 11, 'correct': 7} |
| m3-oracle-tight-1.7B | babilong:qa7 | 256k | {'ok': 100} | 3.0 [0.0, 7.0] | 36.0 [27.0, 45.0] | 1.0 | 0.389 | 4.725 | {'wrong': 77, 'format': 20, 'correct': 3} |
| m3-oracle-tight-1.7B | babilong:qa8 | 8k | {'ok': 100} | 44.0 [34.0, 54.0] | 44.0 [34.0, 54.0] | 1.0 | 0.368 | 4.158 | {'wrong': 56, 'correct': 44} |
| m3-oracle-tight-1.7B | babilong:qa8 | 16k | {'ok': 100} | 47.0 [37.0, 57.0] | 47.0 [37.0, 57.0] | 1.0 | 0.411 | 4.57 | {'wrong': 52, 'correct': 47, 'format': 1} |
| m3-oracle-tight-1.7B | babilong:qa8 | 32k | {'ok': 100} | 40.0 [31.0, 50.0] | 40.0 [31.0, 50.0] | 1.0 | 0.445 | 4.883 | {'wrong': 58, 'correct': 40, 'format': 2} |
| m3-oracle-tight-1.7B | babilong:qa8 | 64k | {'ok': 100} | 44.0 [34.0, 54.0] | 44.0 [34.0, 54.0] | 1.0 | 0.486 | 5.085 | {'wrong': 54, 'format': 2, 'correct': 44} |
| m3-oracle-tight-1.7B | babilong:qa8 | 128k | {'ok': 100} | 44.0 [34.0, 54.0] | 44.0 [34.0, 54.0] | 1.0 | 0.53 | 5.332 | {'wrong': 54, 'correct': 44, 'format': 2} |
| m3-oracle-tight-1.7B | babilong:qa8 | 256k | {'ok': 100} | 37.0 [27.0, 47.0] | 37.0 [27.0, 47.0] | 1.0 | 0.536 | 5.353 | {'wrong': 60, 'correct': 37, 'format': 3} |
| m3-oracle-tight-1.7B | babilong:qa9 | 8k | {'ok': 100} | 83.0 [76.0, 90.0] | 83.0 [76.0, 90.0] | 1.0 | 0.23 | 3.697 | {'correct': 83, 'wrong': 17} |
| m3-oracle-tight-1.7B | babilong:qa9 | 16k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.245 | 3.733 | {'correct': 82, 'wrong': 18} |
| m3-oracle-tight-1.7B | babilong:qa9 | 32k | {'ok': 100} | 81.0 [73.0, 88.0] | 81.0 [73.0, 88.0] | 1.0 | 0.261 | 3.761 | {'correct': 81, 'wrong': 19} |
| m3-oracle-tight-1.7B | babilong:qa9 | 64k | {'ok': 100} | 82.0 [74.0, 89.0] | 82.0 [74.0, 89.0] | 1.0 | 0.279 | 3.766 | {'wrong': 18, 'correct': 82} |
| m3-oracle-tight-1.7B | babilong:qa9 | 128k | {'ok': 100} | 86.0 [79.0, 92.0] | 86.0 [79.0, 92.0] | 1.0 | 0.275 | 3.77 | {'correct': 86, 'wrong': 14} |
| m3-oracle-tight-1.7B | babilong:qa9 | 256k | {'ok': 100} | 85.0 [78.0, 92.0] | 85.0 [78.0, 92.0] | 1.0 | 0.279 | 3.775 | {'correct': 85, 'wrong': 15} |
| m3-oracle-tight-1.7B | infbench:kv_retrieval | full | {'ok': 100} | 55.0 [45.0, 65.0] | – | 1.0 | 3.084 | 3.36 | {'wrong': 45, 'correct': 55} |
| m3-oracle-tight-1.7B | infbench:number_string | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 1.737 | 3.335 | {'correct': 100} |
| m3-oracle-tight-1.7B | infbench:passkey | full | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.969 | 3.335 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 8k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.53 | 3.326 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 16k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.538 | 3.328 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 32k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.547 | 3.324 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.581 | 3.324 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.626 | 3.334 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S1 | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.564 | 3.322 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 8k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.526 | 3.329 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 16k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.537 | 3.327 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 32k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.549 | 3.324 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 64k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.575 | 3.327 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 128k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.632 | 3.328 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multikey | 256k | {'ok': 100} | 100.0 [100.0, 100.0] | – | 1.0 | 0.551 | 3.322 | {'correct': 100} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 8k | {'ok': 100} | 90.0 [84.0, 96.0] | – | 1.0 | 2.117 | 3.497 | {'correct': 90, 'wrong': 10} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 16k | {'ok': 100} | 89.0 [82.0, 95.0] | – | 1.0 | 2.143 | 3.493 | {'correct': 89, 'wrong': 11} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 32k | {'ok': 100} | 90.0 [84.0, 96.0] | – | 1.0 | 2.201 | 3.502 | {'correct': 90, 'wrong': 10} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 64k | {'ok': 100} | 96.0 [92.0, 99.0] | – | 1.0 | 2.226 | 3.497 | {'wrong': 4, 'correct': 96} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 128k | {'ok': 100} | 94.0 [89.0, 98.0] | – | 1.0 | 2.254 | 3.498 | {'correct': 94, 'wrong': 6} |
| m3-oracle-tight-1.7B | synth:S2_multivalue | 256k | {'ok': 100} | 95.0 [90.0, 99.0] | – | 1.0 | 2.34 | 3.498 | {'correct': 95, 'wrong': 5} |

## E2

{"gpu_hours": 13.984, "rows_ok": 1000, "plan": {"n_plan": 40, "n_fixed": 30, "cap_hours": 16, "projected_hours_by_n": {"100": 27.22, "80": 22.8, "60": 18.38, "50": 16.16, "40": 13.95, "30": 11.74}, "projected_hours": 13.95, "seconds_per_example": {"synth:S1|64k": 22.41, "synth:S2_multikey|64k": 22.6, "synth:S2_multivalue|64k": 27.86, "babilong:qa1|64k": 21.02, "babilong:qa2|64k": 20.92, "babilong:qa3|64k": 21.18, "babilong:qa4|64k": 19.72, "babilong:qa5|64k": 19.77, "synth:S1|128k": 81.56, "synth:S2_multikey|128k": 81.11, "synth:S2_multivalue|128k": 96.36, "babilong:qa1|128k": 71.17, "babilong:qa2|128k": 74.24, "babilong:qa3|128k": 72.61, "babilong:qa4|128k": 71.48, "babilong:qa5|128k": 72.25, "infbench:passkey|full": 76.14, "infbench:number_string|full": 78.62, "babilong:qa6|64k": 19.82, "babilong:qa7|64k": 19.07, "babilong:qa8|64k": 19.31, "babilong:qa9|64k": 19.61, "babilong:qa10|64k": 19.66, "babilong:qa6|128k": 71.7, "babilong:qa7|128k": 72.55, "babilong:qa8|128k": 74.24, "babilong:qa9|128k": 71.64, "babilong:qa10|128k": 70.26}}, "n_per_cell": {"babilong:qa1|64k": 40, "babilong:qa1|128k": 40, "babilong:qa10|64k": 30, "babilong:qa10|128k": 30, "babilong:qa2|64k": 40, "babilong:qa2|128k": 40, "babilong:qa3|64k": 40, "babilong:qa3|128k": 40, "babilong:qa4|64k": 40, "babilong:qa4|128k": 40, "babilong:qa5|64k": 40, "babilong:qa5|128k": 40, "babilong:qa6|64k": 30, "babilong:qa6|128k": 30, "babilong:qa7|64k": 30, "babilong:qa7|128k": 30, "babilong:qa8|64k": 30, "babilong:qa8|128k": 30, "babilong:qa9|64k": 30, "babilong:qa9|128k": 30, "infbench:number_string|full": 30, "infbench:passkey|full": 30, "synth:S1|64k": 40, "synth:S1|128k": 40, "synth:S2_multikey|64k": 40, "synth:S2_multikey|128k": 40, "synth:S2_multivalue|64k": 40, "synth:S2_multivalue|128k": 40}, "reserved_above_device": {"device_gib": 24.0, "rows": 32, "rows_ok": 1000, "max_allocated_gib": 19.384, "max_reserved_gib": 30.359, "by_cell": {"babilong:qa1|128k": {"rows": 3, "of": 40, "s_mean": 81.7, "s_mean_others": 71.8}, "babilong:qa10|128k": {"rows": 3, "of": 30, "s_mean": 77.6, "s_mean_others": 70.1}, "babilong:qa2|128k": {"rows": 5, "of": 40, "s_mean": 80.1, "s_mean_others": 71.6}, "babilong:qa3|128k": {"rows": 2, "of": 40, "s_mean": 82.4, "s_mean_others": 72.0}, "babilong:qa4|128k": {"rows": 3, "of": 40, "s_mean": 80.6, "s_mean_others": 71.6}, "babilong:qa5|128k": {"rows": 3, "of": 40, "s_mean": 80.5, "s_mean_others": 70.9}, "babilong:qa6|128k": {"rows": 2, "of": 30, "s_mean": 81.1, "s_mean_others": 70.3}, "babilong:qa7|128k": {"rows": 5, "of": 30, "s_mean": 78.9, "s_mean_others": 69.1}, "babilong:qa8|128k": {"rows": 4, "of": 30, "s_mean": 82.9, "s_mean_others": 70.0}, "babilong:qa9|128k": {"rows": 2, "of": 30, "s_mean": 77.1, "s_mean_others": 69.9}}}}

## Checks

{"oracle_rows_missing_evidence": 0, "retrieval_above_oracle": [], "above_native_reference": [{"series": "m3-tight", "family": "S2", "length": "8k", "accuracy": 0.95, "reference": 0.825}, {"series": "m3-tight", "family": "S2", "length": "16k", "accuracy": 0.9450000000000001, "reference": 0.825}, {"series": "m3-tight", "family": "S2", "length": "32k", "accuracy": 0.95, "reference": 0.825}, {"series": "m3-tight", "family": "S2", "length": "64k", "accuracy": 0.98, "reference": 0.825}, {"series": "m3-tight", "family": "S2", "length": "128k", "accuracy": 0.97, "reference": 0.825}, {"series": "m3-tight", "family": "S2", "length": "256k", "accuracy": 0.975, "reference": 0.825}], "reuse_sources_checked": 5977, "reuse_mismatch": 0, "rerank_same_units_as_hybrid": {"pairs": 5600, "same_units": 5600, "same_prompt": 5600, "same_score": 5600}, "reuse_from_superseded_rows": 60, "budget_rerun_vs_a3": {"synth:S2_multivalue|8k": {"pairs": 100, "score_changed": 0, "same_text": 95, "new_extends_old": 5}, "synth:S2_multivalue|16k": {"pairs": 100, "score_changed": 0, "same_text": 92, "new_extends_old": 8}, "synth:S2_multivalue|32k": {"pairs": 100, "score_changed": 0, "same_text": 79, "new_extends_old": 21}}}

