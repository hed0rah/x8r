#!/usr/bin/env python3
"""
fuzz x8r against tiktoken.

generates N inputs across several flavors, counts tokens with both, and
reports any disagreement with a minimal-ish reproducer. exit status is
the number of mismatches.

flavors:
  ascii_print   random printable ascii
  ascii_full    random bytes in 0x09..0x7e (includes whitespace)
  utf8_mixed    ascii with random valid utf-8 code points sprinkled in
  realcode      random spans sliced from this project's own source
  edge          hand-picked pathological strings

usage:
  scripts/fuzz_vs_tiktoken.py [N]      default 500
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
X8R = str(ROOT / "build" / "x8r")
VOCAB = str(ROOT / "vocab" / "cl100k.bin")
ENC = tiktoken.get_encoding("cl100k_base")


def gen_ascii_print(rng: random.Random) -> bytes:
    n = rng.randint(0, 2048)
    return bytes(rng.randint(0x20, 0x7E) for _ in range(n))


def gen_ascii_full(rng: random.Random) -> bytes:
    n = rng.randint(0, 2048)
    pool = bytes(list(range(0x09, 0x0E)) + list(range(0x20, 0x7F)))
    return bytes(rng.choice(pool) for _ in range(n))


def _rand_codepoint(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.6:
        return rng.randint(0x20, 0x7E)
    if r < 0.85:
        return rng.randint(0xA0, 0x07FF)  # 2-byte (skip control/latin supplement start)
    if r < 0.97:
        # 3-byte, skipping UTF-16 surrogate range
        cp = rng.randint(0x0800, 0xFFFD)
        while 0xD800 <= cp <= 0xDFFF:
            cp = rng.randint(0x0800, 0xFFFD)
        return cp
    return rng.randint(0x10000, 0x10FFFF)  # 4-byte


def gen_utf8(rng: random.Random) -> bytes:
    n = rng.randint(0, 512)
    s = "".join(chr(_rand_codepoint(rng)) for _ in range(n))
    return s.encode("utf-8")  # guaranteed valid


def _collect_realcode_pool() -> bytes:
    chunks = []
    for p in (ROOT / "src").glob("*.c"):
        chunks.append(p.read_bytes())
    for p in (ROOT / "include").glob("*.h"):
        chunks.append(p.read_bytes())
    return b"\n".join(chunks)


REALCODE = _collect_realcode_pool()


def gen_realcode(rng: random.Random) -> bytes:
    if not REALCODE:
        return b""
    n = rng.randint(0, min(4096, len(REALCODE)))
    if n == 0:
        return b""
    start = rng.randint(0, len(REALCODE) - n)
    return REALCODE[start:start + n]


EDGE_CASES = [
    b"",
    b" ",
    b"\n",
    b"\n\n\n",
    b"   ",
    b"a",
    b"'s",
    b"'S",
    b"'RE",
    b"''",
    b"a" * 1000,
    b"  \n  \n  ",
    b"hello world",
    b" hello",
    b"hello ",
    b"hello   ",
    b"123",
    b"1234",
    b"12345",
    b"1,000,000",
    b"!!!",
    b" !!!",
    b"a1b2c3",
    b"\t\n\r\v\f ",
    b"the quick brown fox",
    b"e.g. i.e. etc.",
    b"don't won't can't",
]


def gen_edge(rng: random.Random, i: int) -> bytes:
    return EDGE_CASES[i % len(EDGE_CASES)]


def count_x8r(data: bytes) -> int:
    env = dict(os.environ, X8R_VOCAB=VOCAB)
    r = subprocess.run([X8R, "--count", "-"], input=data, capture_output=True, env=env, check=True)
    return int(r.stdout.strip())


def count_tiktoken(data: bytes) -> int:
    # mirror what our cli does: decode as utf-8 with replacement
    text = data.decode("utf-8", errors="replace")
    return len(ENC.encode_ordinary(text))


def shrink(data: bytes, fail_predicate, keep_utf8: bool = False) -> bytes:
    """simple bisection shrinker. if keep_utf8, only accept candidates
    that are still valid utf-8 so reproducers remain readable."""
    def ok(b: bytes) -> bool:
        if not b:
            return False
        if keep_utf8:
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                return False
        try:
            return fail_predicate(b)
        except Exception:
            return False

    cur = data
    changed = True
    while changed and len(cur) > 1:
        changed = False
        mid = len(cur) // 2
        for cand in (cur[:mid], cur[mid:], cur[:-1], cur[1:]):
            if ok(cand):
                cur = cand
                changed = True
                break
    return cur


def run(n: int, seed: int):
    rng = random.Random(seed)
    generators = [
        ("ascii_print", lambda i: gen_ascii_print(rng)),
        ("ascii_full",  lambda i: gen_ascii_full(rng)),
        ("utf8_mixed",  lambda i: gen_utf8(rng)),
        ("realcode",    lambda i: gen_realcode(rng)),
        ("edge",        lambda i: gen_edge(rng, i)),
    ]

    counts = {k: [0, 0] for k, _ in generators}  # [total, mismatches]
    failures = []

    for i in range(n):
        flavor, gen = generators[i % len(generators)]
        data = gen(i)
        counts[flavor][0] += 1
        try:
            a = count_x8r(data)
            b = count_tiktoken(data)
        except subprocess.CalledProcessError as e:
            failures.append((flavor, data, f"x8r crashed: {e.stderr!r}"))
            counts[flavor][1] += 1
            continue
        if a != b:
            counts[flavor][1] += 1
            def predicate(d, _a=a, _b=b):
                try:
                    return count_x8r(d) != count_tiktoken(d)
                except Exception:
                    return False
            shrunk = shrink(data, predicate, keep_utf8=(flavor in ("utf8_mixed", "realcode")))
            sa = count_x8r(shrunk)
            sb = count_tiktoken(shrunk)
            failures.append((flavor, shrunk, f"x8r={sa} tiktoken={sb}"))

    print("flavor         total  mismatches")
    for k, (t, m) in counts.items():
        print(f"  {k:<12} {t:>6} {m:>11}")
    print()
    if failures:
        print(f"{len(failures)} mismatches. first 20 (shrunk):")
        for flavor, data, msg in failures[:20]:
            preview = data if len(data) <= 80 else data[:77] + b"..."
            print(f"  [{flavor}] len={len(data)} {msg}")
            print(f"    bytes: {preview!r}")
    else:
        print("no mismatches.")
    return sum(m for _, m in counts.values())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0xC0FFEE
    rc = run(n, seed)
    sys.exit(min(rc, 127))
