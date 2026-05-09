# BPE merge loop: priority queue strategies in production tokenizers

Reference notes for the heap implementation in `src/bpe.c`. Compares the
patterns used by tiktoken, HuggingFace tokenizers, and SentencePiece, and
records the design choices made for x8r.

## Why a heap

The merge loop runs once per merge, O(n) times per pre-token. A linear
scan over the live-list is O(n) per find, giving O(n²) total. For pre-tokens
of a few dozen bytes this is fine -- the cache-friendly scan beats any heap
on constant factors. But pre-tokens that hit hundreds or thousands of bytes
(continuous CJK runs, long Latin words, runs of punctuation) make the
quadratic dominant. A min-heap of (rank, idx) pairs drops the find to
O(log n), giving O(n log n) total.

x8r profiling on the stressor corpus shows linear-scan BPE at ~80% of
runtime on `cjk_dense.txt` (13ms of 17ms wall). With the heap path, BPE
drops to ~6ms wall on the same file -- a 13x speedup on that workload.

## Three production implementations

### tiktoken (Rust, OpenAI)

`src/lib.rs` `byte_pair_encode`: dual-path. Pre-tokens of length 1 do
direct lookup. Length 2-99 use a vector linear scan (`_byte_pair_merge`).
Length >= 100 use a `BinaryHeap` + `Vec<State>` (`_byte_pair_merge_large`).
The 100-byte threshold is a comment-justified guess based on cache
locality: "n is often very small so considerations like cache-locality
outweigh the algorithmic complexity downsides of the parts vector."

Staleness: each `State[i]` has a `next_rank` field (rank of the pair
starting at byte i). Heap entries snapshot the rank at push time; on
pop, the heap rank is compared against `state[i].next_rank`. Mismatch =
stale, skip. After a merge, the consumed token's `next_rank` is set to
`Rank::MAX` to invalidate any heap entries referencing it.

PR #442 proposes replacing rank-comparison staleness with explicit
generation counters (`Vec<u32> ver`). Heap entries carry `(rank, idx,
ver_at_push)`; on pop, compare against `ver[idx]`. Strictly more robust
than rank comparison (cannot collide), at the cost of an extra word per
entry. Reports 37-190% throughput gains on multilingual o200k.

### HuggingFace tokenizers (Rust)

Two distinct algorithms for encoding vs training.

**Encoding** (`models/bpe/word.rs` `merge_all`): `dary_heap::QuaternaryHeap`
ordered by merge rank. Word state is a doubly-linked list of `Symbol`
structs with `id`, `len`, `prev`, `next` fields. Staleness check on pop:
1. `symbols[top.pos].len == 0` -- the symbol was consumed (dead-flag).
2. The actual pair at `(symbols[pos], symbols[next])` no longer maps to
   the same merge ID.

The `len == 0` test is more robust than tiktoken's rank comparison
because it uses an explicit boolean signal rather than a value that
could coincidentally match.

**Training** (`models/bpe/trainer.rs` `do_train`): `OctonaryHeap` ordered
by pair frequency. Staleness handled with **refresh-and-repush**:

```rust
let Some(mut top) = queue.pop() else { break; };
if top.count != pair_counts[&top.pair] as u64 {
    top.count = pair_counts[&top.pair] as u64;  // refresh
    queue.push(top);                              // re-push
    continue;
}
```

This pattern is structurally drain-immune: every stale pop on a still-
live entry produces a refreshed push, so the heap size doesn't shrink
faster than valid merges advance.

PR #1618 (`dary_heap` adoption) reports a 9.5% encoding speedup on
multilingual corpus, attributed to fewer cache levels in the heap tree.
Note: this is a small constant-factor gain, not relevant until the heap
is correct and proven to be the bottleneck.

### SentencePiece (C++, Google)

**Encoding** (`bpe_model.cc` `SampleEncode`): `std::priority_queue` of
`SymbolPair*` ordered by score. Staleness check is three-way:

```cpp
if (symbols[top->left].piece.empty() ||
    symbols[top->right].piece.empty() ||
    symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
    continue;
}
```

The size-comparison check (third clause) catches a subtler form of
staleness: when neither symbol has been fully consumed but their
effective boundary has shifted.

**Training** (default path in `bpe_model_trainer.cc`): no heap at all.
Maintains an active set of the top ~5% most frequent bigrams (recomputed
every 100 iterations) and linearly scans that set per merge. PR #1208
adds an opt-in heap path with `HeapDirty` accumulator (lazy correction
applied on pop), reporting 10-24x training speedups.

## Convergence

All three implementations converge on the same architectural skeleton:

1. **Standard heap** (binary, 4-ary, or 8-ary) -- not Fibonacci or
   pairing heap. Only `push`/`pop` are needed; no `decrease-key`.
2. **Lazy deletion** -- never remove or update heap entries in place.
   Stale entries are detected on pop and either skipped or refreshed.
3. **Position-indexed state array** -- authoritative source of merge
   priorities, indexed by stable slot. Slots are marked dead but never
   recycled, so heap-pop staleness checks are safe.
4. **Doubly-linked neighbor pointers** -- O(1) merge updates without
   index recomputation.

Two real choice points:

- **Skip-stale vs refresh-and-repush**. Skip-stale (tiktoken encoding,
  HF encoding) is simpler but relies on heap-size dynamics from new
  pushes outpacing stale drops. Refresh-and-repush (HF training) is
  drain-immune: every stale pop produces exactly one push, so the heap
  cannot shrink due to staleness alone.
- **Generation counter vs value comparison**. Counter is collision-free;
  value comparison can theoretically false-positive if two distinct
  states hash to the same value. In practice, BPE ranks are unique per
  vocab so collision is impossible -- both are safe.

## x8r design choices

`src/bpe.c` adopts:

- **Refresh-and-repush** (HF training pattern). Drain-immunity is worth
  the small bookkeeping cost given that the previous heap attempt failed
  due to drain.
- **Dead-flag via `parts[idx].next < 0`**. Cheaper than length comparison
  and matches the existing `next` field semantics.
- **Single binary heap path**. No threshold split for now -- one path
  exercises the heap on every pre-token and lets the threshold sweep
  measure the crossover empirically.
- **Stack heap of 512 entries, malloc fallback** for longer pre-tokens.
  Mirrors the existing parts-buffer pattern.
- **Initial capacity hint = 4 * (len + 1)** to absorb the worst-case
  push count (initial seed + 2 per merge, refresh-and-repush adding no
  net entries).

## Open question: threshold

The single-path heap regresses ~9% on `utf8_prose_small.txt` and
breaks even on the other small-pre-token corpora. tiktoken switches at
100 bytes, but that's their measurement on their codebase.

The threshold for x8r is a measurable question, not an inherited one.
Sweep on each stressor corpus, plot runtime, pick the crossover. See
the next set of work items in `notes/heap-followup.md`.

## Stale-drain bug retrospective

The first heap attempt (research/opt-loop, since deleted) used
skip-stale with a fixed `len * 2` heap allocation. On `cjk_dense.txt`
(2,013-byte pre-token) it produced 1 token instead of 102,989: the
heap drained to 0 mid-encoding because stale skips outpaced new pushes
(~1.88 stale-to-merge ratio on that input). Root cause was capacity
sizing, not the staleness logic, but the failure mode is fragile to
input distribution. Refresh-and-repush makes the failure structurally
impossible.

## Sources

- tiktoken `src/lib.rs`: https://github.com/openai/tiktoken/blob/main/src/lib.rs
- tiktoken PR #442 (generation counters): https://github.com/openai/tiktoken/pull/442
- HF `tokenizers/src/models/bpe/word.rs` (encoding): https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/word.rs
- HF `tokenizers/src/models/bpe/trainer.rs` (training): https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/trainer.rs
- HF PR #1618 (dary_heap): https://github.com/huggingface/tokenizers/pull/1618
- SentencePiece `src/bpe_model.cc`: https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
- SentencePiece PR #1208 (NLCodec fast BPE): https://github.com/google/sentencepiece/pull/1208
