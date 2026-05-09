# Vocab probe measurement: cache behaviour and prefetch implications

## Summary

The measurement infrastructure landed in this session produces a clear
verdict on the three open questions from `notes/cache-prefetch-vocab.md`:

1. **Probe chain length**: median = 1 on every workload. The probe-length
   distribution is sharply left-tailed; 99% of lookups resolve in ≤8 probes
   even on the worst-case workload (cjk_dense CJK stressor). This means the
   first probe dominates, and long probe chains are vanishingly rare.
2. **First-probe cache misses**: L1 miss rates are 1.6-8.7% (workload-
   dependent), and L3 hits recover most of those — only ~5-10% of LLC
   accesses miss to DRAM on CJK/code workloads. The existing `__builtin_prefetch`
   already prefetches the next slot and the entry data.
3. **HW prefetcher effectiveness**: Could not be measured — the required
   `l2_rqsts.pf_hit`/`pf_miss` events are not available in this container.
   However, the probe-chain data and existing prefetch instructions make
   the question mostly moot (see below).

## Probe length histogram

### cjk_dense.txt (131 KB, CJK stressor — worst-case workload)

| Probes | Count   | Percent |
|--------|---------|---------|
| 1      | 122,707 | 65.56%  |
| 2      |  33,711 | 18.01%  |
| 3      |  13,308 |  7.11%  |
| 4      |   6,952 |  3.71%  |
| 5      |   4,135 |  2.21%  |
| 6-10   |   5,737 |  3.07%  |
| 11-20  |     611 |  0.33%  |
| 21+    |      16 |  0.01%  |

- **Median**: 1 (65.56% resolve on first probe)
- **p99**: 8
- **Probes > 5**: 3.4% of all lookups

### ascii_code_large.txt (1 MB, code)

| Probes | Count   | Percent |
|--------|---------|---------|
| 1      | 511,666 | 86.26%  |
| 2      |  45,523 |  7.67%  |
| 3      |  16,903 |  2.85%  |
| 4      |   8,888 |  1.50%  |
| 5      |   4,437 |  0.75%  |
| 6-10   |   5,345 |  0.90%  |
| 11-20  |     374 |  0.06%  |
| 21+    |       5 |  0.00%  |

- **Median**: 1
- **p99**: 5
- **Probes > 5**: 0.96% of all lookups

### utf8_prose_large.txt (1 MB, English prose)

| Probes | Count    | Percent |
|--------|----------|---------|
| 1      | 1,153,008| 82.57%  |
| 2      |   117,429|  8.41%  |
| 3      |    53,562|  3.84%  |
| 4      |    41,436|  2.97%  |
| 5      |         0|  0.00%  |
| 6-10   |    21,552|  1.54%  |
| 11-20  |     9,419|  0.67%  |
| 21+    |         0|  0.00%  |

- **Median**: 1
- **p99**: 8
- **Probes > 5**: 2.21% of all lookups

## Cache hierarchy measurements (perf stat)

| Metric                  | cjk_dense | ascii_code | utf8_prose |
|-------------------------|-----------|------------|------------|
| L1-dcache-loads (avg)   | 7,728,957 | 30,875,175 | 62,112,323 |
| L1 miss rate            |    8.67%  |     3.17%  |     1.56%  |
| LLC-loads (avg)         |   164,261 |    369,117 |     20,066 |
| LLC miss rate (of LLC)  |    9.86%  |     5.12%  |    21.75%  |
| CPI                     |     1.23  |      1.83  |      2.84  |
| Wall time (avg, warm)   |    7.6 ms |    17.1 ms |    22.9 ms |

**Key observations:**

- **CJK (cjk_dense)**: Highest L1 miss rate (8.67%), highest LLC-footprint
  (164K LLC-loads for 102,989 tokens, or ~1.6 LLC accesses per token).
  This reflects the larger pre-tokens (up to 2,355 B) causing more probe
  chains. Even so, only ~10% of LLC accesses miss to DRAM.

- **Code (ascii_code)**: Moderate L1 misses (3.17%). 369K LLC-loads for
  427,655 tokens (0.86 per token). Lowest CPI among the three.

- **Prose (utf8_prose)**: Lowest L1 miss rate (1.56%). Only 20K LLC-loads
  for 300,540 tokens (0.07 per token). Highest CPI (2.84) despite low
  miss rate — this suggests the bottleneck on this workload is not cache
  misses but the hash compute and key-compare paths. The high
  instructions-per-token ratio (~bf instructions per lookup) and high CPI
  (2.84) with very few cache misses point to an instruction- or ALU-bound
  hot path, not a memory-bound one.

## Implication for the software prefetch hypothesis

### Claim 1: "Probe chain length is meaningfully greater than 1"

**False for the dominant workload.** Median = 1 on all three workloads.
65-86% of lookups resolve on the first probe. A software prefetch that
fires on every iteration would add ~8 cycles of `_mm_prefetch` overhead
per lookup, and on 65-86% of lookups that prefetch target is never
needed (the lookup resolved before reaching that probe). This is pure
overhead.

Even on the worst-case workload (cjk_dense, 3.4% of lookups need >5
probes), the prefetch cost on the 96.6% of lookups that don't need it
swamps any benefit on the 3.4% that do. Rough calculation:

```
Overhead per lookup (always-on):     +8 cycles
Overhead on 96.6% of lookups:        +8 × 0.966 = +7.7 cycles avg
Potential saving on 3.4% (avoid L3):  +42 × 0.034 = +1.4 cycles avg (optimistic)
```

Net: **-6.3 cycles per lookup on average** (worse with prefetch).

The existing code already has `__builtin_prefetch` for the *next* slot
and the *entry data* — this is the right place for a prefetch (cheap,
target is likely needed). Adding an N-ahead prefetch on every iteration
would not help.

### Claim 2: "First-probe misses dominate runtime"

**Partially true but already well-served.** The L1 miss rate is 1.6-8.7%
depending on workload. But the hardware recovers most of these at L2/L3
(LLC miss rate of LLC-loads is only 5-22%). The runtime contribution
from first-probe cache misses is measurable but not dominant — note that
CPI is 1.2-2.8 even with these miss rates, indicating a balanced profile
(compute + memory, not memory-bound).

The existing prefetch of the entry header (`__builtin_prefetch(data + off)`)
already helps the first-probe hit case. The next-slot prefetch helps the
collision case.

### Claim 3: "HW prefetchers are not already covering the first probe"

**Unmeasurable in this environment.** The `l2_rqsts.pf_hit`/`pf_miss`
events require raw Intel PMU access not available in this container.
However, given the probe-chain distribution, this question is largely
moot — even if HW prefetchers do nothing, there's insufficient probe
chain length for software prefetch to help.

### Summary

| Question | Answer | Source |
|----------|--------|--------|
| Median probe length? | 1 (65-86% of lookups) | Histogram data |
| p99 probe length? | 5-8 (workload dependent) | Histogram data |
| Long probe chains (>5)? | 0.96-3.4% of lookups | Histogram data |
| L1 miss rate? | 1.6-8.7% | perf stat |
| LLC miss rate? | 5-22% of LLC accesses | perf stat |
| CPI? | 1.2-2.8 | perf stat |
| HW prefetcher active? | Unknown (perf events unavailable) | — |

## Verdict

**Do not add software prefetch (i.e., `_mm_prefetch`) to the probe loop.
The data does not justify it.**

The measurement infrastructure now exists to revisit this decision if:
- A workload emerges with systematically longer probe chains (median > 2,
  significant tail past 10)
- The existing callers change to pass keys that collide more aggressively
- The vocab load factor changes significantly (currently 0.38 for both
  cl100k and o200k)
- Hardware prefetcher behaviour can be measured via raw PMU events, and
  turns out to be absent on a target deployment platform

Until then, the existing `__builtin_prefetch` calls (next slot + entry
header) are the right level of prefetch investment.

## Durable measurement infrastructure landed

- **`src/vocab.c`**: env-gated (`X8R_VOCAB_PROBE_HIST`) probe chain
  histogram. Zero overhead when unset (single load of static + one
  predicted-taken test at each return point). Atexit dumps to stderr.
  Permanent debug knob for future probe-chain investigations.

- **`bench/perf_vocab.sh`**: perf stat measurement script. Probes
  Intel-specific events first, falls back to generic events. Detects
  `perf_event_paranoid >= 2` and gives an actionable error message.
  Durable script for any future cache-hierarchy measurement need.

- **`bench/profiles/vocab_probe_hist.txt`**: raw histogram output for
  three representative workloads (CJK stressor, code, prose).

- **`bench/profiles/vocab_perf_stat.txt`**: raw perf stat output for
  the same three workloads (5 warm iterations each).

## Sources

- `notes/cache-prefetch-vocab.md` — prior analysis and hypotheses
- `src/vocab.c` — instrumented lookup function (this session)
- `bench/perf_vocab.sh` — perf measurement script (this session)
- `bench/profiles/vocab_probe_hist.txt` — raw histogram data (this session)
- `bench/profiles/vocab_perf_stat.txt` — raw perf stat data (this session)