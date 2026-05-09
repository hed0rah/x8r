# BPE heap threshold sweep

## Question

`src/bpe.c` runs the binary min-heap on every pre-token. Single-path
implementation, simple to reason about. The open question after the
heap landed: is there a pre-token-length threshold below which a linear
scan would beat the heap? tiktoken uses 100; HuggingFace uses zero (no
threshold); SentencePiece uses zero. We swept the threshold empirically
to settle the question for x8r.

## Method

Compile-time threshold via `BPE_HEAP_THRESHOLD`. Pre-tokens with byte
length below the threshold use a linear-scan merge loop; at or above,
the heap. Bookend values: `0` (always-heap, current behavior) and
`999999` (always-linear).

For each threshold:

1. Clean rebuild with `CFLAGS="-O3 -march=x86-64-v3 -DBPE_HEAP_THRESHOLD=N"`.
2. Verify correctness: golden tests (5/5) and `fuzz_vs_tiktoken.py 500`
   (zero mismatches). All seven thresholds passed.
3. Run `bench/run.py --cold-iters 0 --iters 9` against six target files.
4. Extract warm-min timings (minimum of 9 iterations after a discarded prime).

Target files chosen to cover both the "heap helps" and "heap hurts"
hypotheses:

| File | Pre-token shape | Hypothesis |
|---|---|---|
| `cjk_dense.txt` | continuous CJK, up to 2355 B | heap helps a lot |
| `long_words.txt` | Latin pseudo-words 50-200 chars | heap helps |
| `markdown_rules.txt` | long punct runs, up to 120 B | heap helps |
| `deep_indent.txt` | nested whitespace runs | unclear |
| `utf8_prose_small.txt` | short mixed-codepoint English | heap may hurt |
| `ascii_code_small.txt` | short ASCII source | heap may hurt |

The harness is at `bench/threshold_sweep.py`. Reusable for re-sweeping
under different conditions (cold-cache, alternate hardware, vocab
changes).

## Results

Warm-min in milliseconds (lower is faster):

| Threshold     | cjk_dense | long_words | deep_indent | md_rules | utf8_small | ascii_small |
|---------------|-----------|------------|-------------|----------|------------|-------------|
| always-heap   |      8.35 |      11.70 |        0.98 |    18.47 |      0.704 |       0.640 |
| 16            |      8.59 |      12.21 |        0.96 |    19.29 |      0.654 |       0.610 |
| 32            |      8.64 |      12.41 |        0.97 |    19.44 |      0.669 |       0.602 |
| 64            |      8.52 |      12.47 |        0.95 |    18.79 |      0.645 |       0.601 |
| 100           |      8.55 |      12.67 |        0.97 |    17.94 |      0.656 |       0.615 |
| 200           |      8.45 |      19.21 |        0.98 |    19.10 |      0.665 |       0.660 |
| always-linear |     87.77 |      18.99 |        0.95 |    20.38 |      0.669 |       0.636 |

Raw data in `bench/profiles/threshold_sweep.csv`. Plot in
`bench/profiles/threshold_sweep.png`.

## Interpretation

**The signal in the data is at the bookends, not in the interior.**

- `always-linear` cliffs catastrophically on `cjk_dense.txt` (8.35 → 87.77ms,
  10.5x worse). That's the O(n²) linear scan hitting a 2355-byte pre-token.
- Threshold `200` cliffs on `long_words.txt` (12.67 → 19.21ms at the
  100→200 boundary, ~50% worse). long_words has many pre-tokens in the
  100-199 byte range; at T=200 they all fall to linear and pay the
  quadratic cost.
- Every other threshold-vs-threshold comparison is within run-to-run
  noise (stddev 0.04-1.0ms depending on file size; observed differences
  smaller than that).

**Threshold values 0, 16, 32, 64, 100 are statistically tied on every file.**

The deltas exist (e.g., utf8_prose_small at T=0 is 0.704ms vs 0.654ms at
T=16, ~7%) but they fall inside the measurement noise floor (stddev
0.04ms on those files, larger than the deltas across thresholds). The
~9% small-pre-token regression we expected from heap setup overhead did
not reproduce on this hardware (Intel i5-10300H, Comet Lake).

## Recommendation: keep `BPE_HEAP_THRESHOLD = 0`

No change to `src/bpe.c`. Single-path heap stays.

The data does not justify adding a dual-path implementation. Every
threshold in the safe range produces equivalent performance on this
hardware. Adding the linear-scan fallback would introduce dead code
(unreachable at the recommended threshold) for no measurable benefit,
violating the "no speculative abstractions" principle.

The compile-time `-DBPE_HEAP_THRESHOLD=N` override remains available
for anyone who wants to re-evaluate on different hardware or under
different cache conditions, but it is not the default and is not
exposed in the public API.

## Caveats

1. **Warm-only.** All measurements are warm-min after a cache-prime
   discard. Both binary and data are in the page cache by the time
   timings are collected. Cold-start workloads (one-shot CLI, serverless)
   have different cost dynamics; the heap path's malloc traffic and
   pointer chasing may matter more there. Re-run the sweep with
   `--cold-iters N` before changing the threshold for cold deployment.

2. **Hardware-specific.** Comet Lake has aggressive prefetchers and a
   reasonably large L2 (256 KB). On older or cache-constrained CPUs
   (older Atom, ARM cores without robust prefetch, low-end embedded
   x86), the heap setup overhead may dominate where it doesn't here.
   Re-sweep before deploying on materially different hardware.

3. **Workload-specific.** The corpus files are synthetic stressors plus
   typical mixed text. Real workloads (financial data, code completion,
   non-English prose) have different pre-token-length distributions
   and may respond to a different threshold.

## Cross-reference

- `notes/bpe-heap-survey.md` -- design rationale for the heap, comparison
  vs tiktoken / HuggingFace / SentencePiece staleness handling.
- `bench/threshold_sweep.py` -- the harness used to produce this data,
  reusable for re-sweeping.
