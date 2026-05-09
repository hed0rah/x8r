# x8r dev cheatsheet

quick reference for running tests, benchmarks, and validating changes.

## build

```sh
make -j$(nproc)              # build everything
make -B -j$(nproc) -s        # force rebuild, silent
make clean                   # nuke build/
```

binaries land in `build/`:
- `build/x8r` -- CLI (`./build/x8r --count <file>`)
- `build/libx8r.so`, `build/libx8r.a` -- library

## venv setup

most python helpers want tiktoken installed.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken
```

`.venv/` is gitignored. activate it before running any benchmark or fuzz
script.

## tests

### golden (5 hand-curated cases, fast)

```sh
bash tests/run_golden.sh
```

uses cl100k by default. set `X8R_VOCAB=./vocab/o200k.bin` to test o200k.

### bit-exact corpus comparison vs tiktoken (16 files × 2 vocabs = 32 cases)

```sh
source .venv/bin/activate
python3 - <<'EOF'
import tiktoken, subprocess, os
from pathlib import Path
ROOT = Path('.').resolve()
fails = ok = 0
for model in ('cl100k', 'o200k'):
    enc = tiktoken.get_encoding(f'{model}_base')
    env = dict(os.environ, X8R_VOCAB=str(ROOT/'vocab'/f'{model}.bin'))
    for f in sorted((ROOT/'bench'/'corpus').iterdir()):
        if f.is_dir(): continue
        text = f.read_bytes().decode('utf-8')
        exp = len(enc.encode_ordinary(text))
        got = int(subprocess.run([str(ROOT/'build'/'x8r'),'--count',str(f)],
            capture_output=True, env=env, check=True).stdout.strip())
        ok += got == exp; fails += got != exp
        print(('OK  ' if got == exp else 'FAIL'), f'{model:6s}', f.name, got, exp)
print(f'{ok} pass, {fails} fail')
EOF
```

### fuzz vs tiktoken (5000 random cases per vocab, ~30s)

```sh
source .venv/bin/activate
python3 scripts/fuzz_vs_tiktoken.py 5000               # cl100k
X8R_FUZZ_MODEL=o200k python3 scripts/fuzz_vs_tiktoken.py 5000   # o200k
```

prints per-flavor mismatch count. exit code = number of mismatches.

## corpus

regenerate the 16 corpus files (12 base + 4 stressors):

```sh
python3 bench/gen_corpus.py
```

stressor files exercise long pre-tokens that the base corpus doesn't reach:
- `cjk_dense.txt` -- continuous `\p{L}+` CJK, up to 2355 B per pre-token
- `long_words.txt` -- 50-200 char Latin pseudo-words
- `deep_indent.txt` -- long `\s+` runs from nested indentation
- `markdown_rules.txt` -- long `[^\s\p{L}\p{N}]+` punct runs

## benchmark

```sh
source .venv/bin/activate
python3 bench/run.py --iters 7 --cold-iters 0 --json-out bench/results.json
```

flags:
- `--iters N` -- warm-run iterations (default 7, discards first)
- `--cold-iters N` -- cold-run iterations (default 5, requires duncache)
- `--cold-iters 0` -- skip cold runs (no duncache needed)
- `--only PATTERN` -- glob filter on corpus filenames
- `X8R_BENCH_MODEL=o200k` -- benchmark against o200k vocab (default cl100k)

cold runs need `/tmp/duncache/duncache` (page-cache eviction tool). symlink
it from a separate clone of `getcompanion-ai/duncache` if you want cold
numbers. without it, use `--cold-iters 0`.

output: ratio table at the end (`spd_warm > 1` means x8r faster than
tiktoken).

## profile

flamegraph generation lives outside the repo. quick perf check:

```sh
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses \
    ./build/x8r --count bench/corpus/cjk_dense.txt
```

## vocab

regenerate vocab blobs (rare, only if vocab format changes):

```sh
python3 scripts/dump_cl100k.py     # -> vocab/cl100k.bin
python3 scripts/dump_o200k.py      # -> vocab/o200k.bin
```

## quick smoke test before any commit

```sh
make -j$(nproc) -s && \
bash tests/run_golden.sh && \
source .venv/bin/activate && \
python3 scripts/fuzz_vs_tiktoken.py 1000 && \
X8R_FUZZ_MODEL=o200k python3 scripts/fuzz_vs_tiktoken.py 1000
```

if all four pass, the change is safe to commit.

## bpe heap notes

`src/bpe.c` uses a binary min-heap with refresh-and-repush staleness
handling. the critical correctness gate is `cjk_dense.txt -> 102,989`
tokens on cl100k. design rationale and survey of tiktoken / huggingface /
sentencepiece patterns lives in `notes/bpe-heap-survey.md`.
