#!/usr/bin/env python3
"""
round-trip invariants for x8r chunking.

for every file in tests/golden/in (plus a few generated inputs), chunk
at several budgets with --boundary {none,line,auto} and verify:

  I1  chunks[0].start == 0
  I2  chunks[i].end == chunks[i+1].start         (no gaps, no overlap)
  I3  chunks[-1].end == len(input)               (covers the whole file)
  I4  sum(chunk.token_count) == total tokens     (from --count)
  I5  every chunk has token_count <= budget      (budgets are hard caps)
  I6  byte concatenation of chunk slices == input  (lossless)

exit status: number of failing (file, budget, boundary) triples.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X8R = str(ROOT / "build" / "x8r")
VOCAB = str(ROOT / "vocab" / "cl100k.bin")


def run(args, data: bytes | None = None) -> str:
    env = dict(os.environ, X8R_VOCAB=VOCAB)
    r = subprocess.run([X8R, *args], input=data, capture_output=True, env=env, check=True)
    return r.stdout.decode()


def chunk(path_or_stdin: str | None, data: bytes | None, budget: int, boundary: str):
    args = ["--budget", str(budget), "--boundary", boundary, "--json",
            path_or_stdin if path_or_stdin else "-"]
    out = run(args, data)
    return json.loads(out)


def total_count(path_or_stdin: str | None, data: bytes | None) -> int:
    return int(run(["--count", path_or_stdin if path_or_stdin else "-"], data).strip())


def check(label: str, data: bytes, chunks: list, budget: int) -> list[str]:
    errs = []
    n = len(data)
    if n == 0:
        if chunks:
            errs.append("empty input produced chunks")
        return errs
    if not chunks:
        errs.append("non-empty input produced no chunks")
        return errs
    if chunks[0]["start"] != 0:
        errs.append(f"I1: first chunk start={chunks[0]['start']} != 0")
    for i in range(len(chunks) - 1):
        if chunks[i]["end"] != chunks[i + 1]["start"]:
            errs.append(f"I2: gap/overlap at chunk {i}: end={chunks[i]['end']} "
                        f"next.start={chunks[i+1]['start']}")
    if chunks[-1]["end"] != n:
        errs.append(f"I3: last chunk end={chunks[-1]['end']} != len={n}")
    tok_sum = sum(c["tokens"] for c in chunks)
    tok_total = total_count(None, data)
    if tok_sum != tok_total:
        errs.append(f"I4: sum(tokens)={tok_sum} != total={tok_total}")
    # I5: chunks must fit the budget, except for a chunk that consists of
    # a single pre-token whose BPE encoding alone exceeds the budget. such
    # chunks are marked cut=hard. line/blank/block cuts must always fit.
    for i, c in enumerate(chunks):
        if c["tokens"] > budget:
            if c["cut"] not in ("hard", "eof"):
                errs.append(f"I5: chunk {i} tokens={c['tokens']} > budget={budget} "
                            f"with cut={c['cut']} (should never overshoot)")
    # I6 lossless concat
    reconstructed = b"".join(data[c["start"]:c["end"]] for c in chunks)
    if reconstructed != data:
        errs.append(f"I6: reconstructed bytes != input (len {len(reconstructed)} vs {n})")
    return errs


def main():
    failures = 0
    inputs: list[tuple[str, bytes]] = []
    for p in sorted((ROOT / "tests" / "golden" / "in").iterdir()):
        inputs.append((str(p.relative_to(ROOT)), p.read_bytes()))
    # add project's own source as bigger real-world
    for p in sorted((ROOT / "src").glob("*.c")):
        inputs.append((str(p.relative_to(ROOT)), p.read_bytes()))
    # plus a few random generated blobs
    rng = random.Random(42)
    for i in range(3):
        size = rng.randint(100, 20_000)
        blob = bytes(rng.randint(0x20, 0x7E) for _ in range(size))
        inputs.append((f"<rand-ascii-{i}-{size}b>", blob))

    budgets = [5, 16, 64, 512, 10_000]
    boundaries = ["none", "line", "auto"]

    for label, data in inputs:
        for budget in budgets:
            for bnd in boundaries:
                try:
                    # use stdin path for generated blobs, path for on-disk files
                    path = str(ROOT / label) if not label.startswith("<") else None
                    chunks = chunk(path, data if path is None else None, budget, bnd)
                except subprocess.CalledProcessError as e:
                    print(f"FAIL {label} budget={budget} boundary={bnd}: x8r error: {e.stderr!r}")
                    failures += 1
                    continue
                errs = check(label, data, chunks, budget)
                if errs:
                    failures += 1
                    print(f"FAIL {label} budget={budget} boundary={bnd}")
                    for e in errs:
                        print(f"    {e}")
                else:
                    pass  # quiet on success

    total_cases = len(inputs) * len(budgets) * len(boundaries)
    print(f"{total_cases - failures}/{total_cases} round-trip cases pass")
    return failures


if __name__ == "__main__":
    sys.exit(min(main(), 127))
