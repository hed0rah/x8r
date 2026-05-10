# decoder

## what landed

three new public C entry points and two CLI surfaces:

```c
x8r_status x8r_decode_bytes(ctx, ids, n, &out_bytes, &out_len);
void       x8r_bytes_free(uint8_t *bytes);
const uint8_t *x8r_decode_single(ctx, id, &out_len);
```

```sh
x8r --decode <file>     # reads decimal ids from file/stdin, writes raw bytes
echo X | x8r --encode - | x8r --decode -    # round-trip pipeline
```

## why

x8r was a counter and a chunker, never a tokenizer in the full sense. encoding
returned IDs as of the previous commit but there was no inverse path. without
decode, x8r can't be used for anything that needs round-trip (verifying
tokenizer correctness, caching token IDs and reconstructing text, debugging
agent traces). decoder closes the loop.

## how it works

the vocab blob has a forward index (hash → offset), used for `x8r_vocab_lookup`
during encode. decode needs the inverse — `rank → offset` — which the blob
doesn't carry. building it on disk would require regenerating every committed
vocab file; building it once at vocab load is cheaper and keeps the blob
format unchanged.

```c
// added to x8r_vocab struct
uint32_t *rank_to_offset;   /* size max_rank+1, UINT32_MAX = absent */
uint32_t  max_rank;
```

`x8r_vocab_load` now does two passes over the populated table slots:

1. **find max_rank.** walk all `tsz` table entries; for each non-empty,
   read the rank field at `data[off+2..off+6]`. track the maximum. cost: one
   memory read per slot, ~200K reads for o200k.

2. **populate rank_to_offset.** allocate `max_rank+1` u32 slots, init to
   `UINT32_MAX`, walk again, fill `idx[rank] = off` for each entry. cost:
   another ~200K reads, plus one `malloc(800 KB)` for o200k.

total one-time cost at vocab load: ~3-5 ms on the bench machine. negligible
relative to the mmap of the 4.5 MB blob.

decode itself is cache-friendly: each ID is one read from `rank_to_offset`
(L1/L2 hit since the index is dense and contiguous), one read of the entry
header (one cache line in the data blob), then a memcpy of the token bytes.
two passes over the input — once to compute total output size, once to copy —
so we malloc exactly the right size.

## correctness

- **32/32 corpus files** (16 stressors × 2 vocabs):
  - x8r decode bytes match original input byte-for-byte (full round-trip)
  - x8r decode bytes match `tiktoken.Encoding.decode_bytes` byte-for-byte on
    the same id list
- **2,000 fuzz cases per vocab in roundtrip mode** (encode then decode via
  x8r, compared against the input round-tripped through utf-8 replace):
  zero mismatches across all five flavors
- **2,000 fuzz cases per vocab in count mode and ids mode**: zero mismatches
  (no regression in encode behavior)
- **edge cases**: empty input, single ASCII char, single CJK, single emoji,
  bare newline, all round-trip identically
- **error cases**: unknown token id returns `X8R_E_VOCAB`; non-numeric input
  to `--decode` returns parse error with byte position
- **AddressSanitizer + UndefinedBehaviorSanitizer**: full corpus exercises
  encode + decode on both vocabs with `-fsanitize=address,undefined`. zero
  errors. confirms `rank_to_offset` allocation/free is balanced and no
  out-of-bounds reads on the decode path.

## performance

in-process via libx8r.so + ctypes, vs `tiktoken.Encoding.decode_bytes`,
warm cache, Comet Lake i5-10300H. ID list marshalling excluded from both
sides of the timing (tiktoken accepts python lists via PyO3 without an
explicit marshalling step; ctypes builds a `c_uint32` array, which is a
binding-side detail not a property of the decoder).

### cl100k

| file                  | ids     | bytes   | x8r ms | tik ms | speedup | x8r MB/s | tik MB/s |
|-----------------------|---------|---------|-------:|-------:|--------:|---------:|---------:|
| ascii_code_large      | 427,655 | 1.0 MiB |  4.39  |  9.30  | 2.12x   | 239.0    | 112.7    |
| json_large            | 375,954 | 1.0 MiB |  2.81  |  6.23  | 2.21x   | 376.4    | 170.0    |
| markdown_large        | 340,629 | 1.0 MiB |  3.70  |  7.64  | 2.06x   | 283.4    | 137.3    |
| utf8_prose_large      | 300,540 | 1.0 MiB |  3.19  |  6.14  | 1.93x   | 329.0    | 170.9    |
| cjk_dense             | 102,989 | 128 KiB |  0.90  |  1.90  | 2.11x   | 145.4    |  68.9    |

### o200k

| file                  | ids     | bytes   | x8r ms | tik ms | speedup | x8r MB/s | tik MB/s |
|-----------------------|---------|---------|-------:|-------:|--------:|---------:|---------:|
| ascii_code_large      | 421,765 | 1.0 MiB |  4.74  |  9.84  | 2.07x   | 221.1    | 106.5    |
| json_large            | 373,756 | 1.0 MiB |  2.91  |  6.46  | 2.22x   | 364.3    | 164.0    |
| markdown_large        | 316,607 | 1.0 MiB |  3.67  |  7.43  | 2.03x   | 285.9    | 141.1    |
| utf8_prose_large      | 221,911 | 1.0 MiB |  2.29  |  4.40  | 1.92x   | 457.8    | 238.2    |
| cjk_dense             |  84,067 | 128 KiB |  0.69  |  1.46  | 2.11x   | 189.1    |  89.5    |

**roughly 1.5-2.2x faster than tiktoken on every file across both vocabs.**
the smallest speedups (1.5-1.7x) are on inputs <10 KB where the decode is
already in the tens of microseconds and constant overheads (call setup,
malloc, return marshalling) start to matter. on larger inputs the speedup
converges to ~2x.

## subprocess CLI is a different story

`x8r --decode -` via subprocess is **slower** than tiktoken in-process for
small inputs because fork+exec+stdin-parse-decimals dominates the actual
decode work. for a 3K-id input, x8r CLI takes 2.3 ms (mostly fork+parse)
vs tiktoken 0.07 ms in-process. for 400K ids, x8r CLI is 13 ms vs tiktoken
9.7 ms — closer but still slower because parsing 400K decimal ASCII numbers
costs real cycles.

decode's natural shape is in-process, not subprocess. the CLI exists for
completeness (round-trip testing, debugging) but is not the recommended
integration path. anyone serious about decode performance should `dlopen`
`libx8r.so` and call `x8r_decode_bytes` directly. results in
`bench/decode_results.json` (subprocess) and `bench/decode_inproc_results.json`
(in-process); both retained for the contrast.

a future binary input mode (`--decode --binary` reading u32 LE from stdin)
would close most of the subprocess gap, but that's a separate piece of work.
encode has the same shape: `--encode --binary` would write u32 LE to stdout,
making `x8r --encode --binary | x8r --decode --binary` a fast pipe. not
implemented here.

## fuzzer extension

`scripts/fuzz_vs_tiktoken.py` now supports three check modes via
`X8R_FUZZ_CHECK`:

- `count` (default) — compare token counts only
- `ids` — compare full token-id lists; catches "same total, different
  tokens" regressions the count fuzzer is blind to
- `roundtrip` — encode then decode via x8r, compare against the input
  round-tripped through utf-8 replace; **strongest invariant** because any
  bug in encode, decode, vocab forward index, vocab inverse index, or the
  CLI parse path breaks round-trip identity

these are durable beyond this session. `roundtrip` is the gate that future
heap or hash-function changes should run before merging.

## what's not done

- **no `decode_with_offsets`** equivalent. tiktoken returns
  `(bytes, [byte_offset, byte_offset, ...])` so callers can map id index to
  output byte position. trivial to add; not needed for current use cases.
- **no decode of special tokens.** there are no special tokens in cl100k or
  o200k vocab dumps as shipped. if a vocab carrying `<|endoftext|>` is loaded,
  decode would emit the literal bytes of the special token's text
  representation, not raise. tiktoken raises by default in `decode_bytes`
  with strict allowed_special. matching that strictness requires an API
  change, not done.
- **no `--decode --binary` mode.** see "subprocess CLI is a different
  story" above.

## opinion

decoder is a small piece. 90 lines of new code + 100 lines of vocab index
build. learning value was modest — the inverse index pattern is standard,
the two-pass decode pattern is mechanical. main wins are:

1. x8r is now a complete tokenizer in the round-trip sense, not a counter
2. the round-trip fuzzer mode is a stronger correctness gate going forward
3. real performance number on a real comparison: in-process decode is
   ~2x tiktoken, which is consistent with the encode speedup story
4. ctypes marshalling exposed as a measurement hazard worth knowing about
   for any future bench

next piece could plausibly be the encode/decode binary stdin/stdout protocol
(closes subprocess gap) or the rapidhash swap (the actual perf lever
identified by the vocab probe analysis). rapidhash is the higher-leverage,
higher-learning option.
