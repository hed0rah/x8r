# Cache miss analysis and software prefetch for `x8r_vocab_lookup`

Reference notes for any future work on the vocab-lookup hot path. Carries
the cache-hierarchy priors, hardware-prefetcher behaviour, and the proposed
software-prefetch hypothesis. The hypothesis is **not** a recommendation --
it is conditional on measurements that have not yet been taken.

## Why the function is interesting

`x8r_vocab_lookup` is 38% of self-time in the v0.3 flamegraph
(`bench/profiles/x8r_v03.svg`), the single hottest function. It does:

1. FNV-1a 32-bit hash over the input bytes (1-6 bytes for vocab keys).
2. Open-addressing linear probe over a flat hash table (cl100k ~2.2 MB,
   o200k ~4.5 MB), 4-byte slots, load 0.38, table size power-of-two.

The table exceeds L1 (32 KB) by 70x and L2 (256 KB) by 8-18x. The first
probe accesses a pseudo-random offset within the table -- almost
guaranteed L1 miss, very likely L2 miss, often L3 hit. Subsequent probes
within the same cache line are cheap (L1 hit). So the cost is dominated
by the **first probe of each lookup**.

## Cache hierarchy on the measurement platform

Comet Lake i5-10300H (Skylake-derived, AVX2 only). Latencies from
uops.info microbenchmarks; sizes confirmed via `/proc/cpuinfo` and
inspection.

| Level | Size       | Assoc | Line | Latency (cycles) | At 2.5 GHz |
|-------|-----------:|:-----:|:----:|------------------|-----------:|
| L1d   | 32 KB      | 8-way | 64 B | 4                | 1.6 ns     |
| L2    | 256 KB     | 4-way | 64 B | 12               | 4.8 ns     |
| L3    | 8 MB       | 16-way| 64 B | ~42              | 16.8 ns    |
| DRAM  | -          | -     | 64 B | ~170             | 68 ns      |

L1 DTLB is 64 entries (covers 256 KB at 4 KB pages -- exactly L2). STLB
1536 entries / 6 MB. cl100k vocab fits in STLB; o200k may overflow under
contention.

**Critical observation for prefetch sizing:** an L3 hit costs 42 cycles
(roughly 10x an L1 hit). One avoidable L3 miss is worth ~10 L1 hits.

## Hardware prefetchers (Intel Skylake-derivative)

Four prefetchers, three of which could plausibly help with a probe loop:

1. **L1 DCU prefetcher** -- detects sequential ascending access. Triggers
   after ~2 consecutive loads. Prefetches 2-3 lines ahead.
2. **L1 DCU IP prefetcher** -- detects fixed-stride loads from a single
   instruction pointer (stride <= 2 KB). Prefetches 8-16 lines ahead.
3. **L2 Streamer** -- detects forward/backward sequential streams within
   a 4 KB page. Tracks 32 streams concurrently. Up to 20 lines ahead.
4. **L2 Spatial** -- always-on; pairs every loaded line with its 128-byte
   partner.

Theoretical analysis of how they interact with the probe loop:

- Within a cache line (probes 2-16): L1 IP prefetcher and L2 Streamer
  are likely active and helping.
- **First probe of each lookup**: random offset, no detectable pattern.
  HW prefetchers should not help. This is the dominant cost.
- Across 4 KB pages: L2 Streamer caps at 32 tracked pages. cl100k
  offset table spans ~200 pages, far exceeding capacity.

**This is a theoretical claim, not a measured one.** The right way to
verify is `perf stat` with `l2_rqsts.pf_hit` and `l2_rqsts.pf_miss`
counters on representative workloads. If the HW prefetchers ARE
firing usefully on x8r's pattern, software prefetch is at best
redundant.

## Hypothesis: software prefetch with `_mm_prefetch`

Insert `_mm_prefetch(&table[(i + N) & mask], _MM_HINT_T1)` ahead of the
probe by N slots, where N is chosen to hide L3 latency:

```
distance = miss_penalty / cycles_per_iter = 42 / 8 = ~5
```

Allowing some headroom and to cover a fraction of DRAM-resident probes,
N = 8-16. Hint T1 (L2 + L3, not L1) avoids polluting L1 with hash table
entries that will only be touched once.

Prefetch instruction cost is ~8 cycles per call on Skylake. The proposed
naive placement fires on every iteration of the probe loop:

```c
for (probe = 0; probe < table_size; ++probe) {
    _mm_prefetch(&table[(i + N) & mask], _MM_HINT_T1);  // every iter
    ...
}
```

## Why the recommendation is not yet justified

Three unmeasured assumptions must hold for the change to win:

1. **Probe chain length is meaningfully greater than 1.** With load
   factor 0.38 and linear probing, the analytical expected probe length
   is ~1.3. If most lookups resolve on the first probe, then per
   lookup the prefetch cost is +8 cycles, the probe-ahead target is
   never reached, and we pay overhead for nothing. *Measurement
   needed: probe-chain length histogram on representative workloads.*

2. **First-probe misses dominate runtime.** The 38% self-time figure is
   wall-clock total for the function; how much of that is cache-miss
   cost vs FNV-1a hash compute vs key-comparison vs branch overhead is
   unknown. *Measurement needed: `perf stat -e mem_load_retired.l1_hit,
   l1_miss, l2_hit, l3_hit, l3_miss` per workload.*

3. **HW prefetchers are not already covering the first probe.** The
   theoretical claim (§3) is plausible but not proven. *Measurement
   needed: `perf stat -e l2_rqsts.all_demand_data_rd, l2_rqsts.pf_hit,
   l2_rqsts.pf_miss, l1d_pend_miss.fb_full`.*

If (1) is false (mean probe length ~= 1), the change is net negative.
If (2) is false (most "self-time" is hash compute), the change buys
little. If (3) is false (HW prefetchers already work), the change is
redundant.

The order to resolve these: (1) is cheapest -- in-process counter,
no perf access needed. (2) and (3) require `perf stat` and may require
unprivileged-perf-events configuration on the host.

## Refinement worth pre-staging

Even if the data justifies a prefetch, the naive every-iteration
placement is suboptimal. Two cheaper variants:

- **Front-load only**: one prefetch on entry to the probe loop, none
  per-iteration. Helps the first probe at no per-iteration cost. Loses
  the staircase effect on long probe chains, but those are rare.
- **Conditional on probe count**: skip prefetch for probes 0-1, fire
  it only on probe >= 2 (where the long-tail latency lives). Requires a
  branch per iteration but cheaper than always-on.

Pick after seeing the histogram.

## Sources

| # | Source | URL |
|---|---|---|
| 1 | uops.info Skylake cache latency | https://uops.info/cache.html |
| 2 | ChipsAndCheese on Skylake STLB | https://chipsandcheese.com/2022/10/14/skylake-intels-longest-serving-architecture/ |
| 3 | Intel Community on the four hardware prefetchers | https://community.intel.com/t5/Software-Tuning-Performance/How-to-control-the-four-hardware-prefetchers-in-L1-and-L2-more/td-p/1104586 |
| 4 | `_mm_prefetch` locality hints | https://stackoverflow.com/questions/46521694/what-are-mm-prefetch-locality-hints |
| 5 | L2 streamer behaviour (32 streams, 20-line distance) | https://community.intel.com/t5/Software-Tuning-Performance/understanding-the-behavior-logic-of-the-L2-stream-prefetcher/td-p/1612944 |
| 6 | Redis dictFind prefetch optimization (PR #13646) | https://github.com/redis/redis/pull/13646 |
| 7 | xxHash issue #485 (prefetch distance / NTA tuning) | https://github.com/Cyan4973/xxHash/issues/485 |

## Cross-reference

- `notes/threshold-sweep.md` -- the immediately prior application of
  the same "measure before changing code" framework on the BPE heap
  threshold question. Same shape: theoretical priors said one thing,
  measurement on real workloads said the priors didn't apply.
