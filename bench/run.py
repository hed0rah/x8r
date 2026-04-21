#!/usr/bin/env python3
"""
benchmark harness: x8r --count vs tiktoken, cold + warm.

cold: evict corpus pages via /tmp/duncache/duncache -q --no-sync before each run.
warm: run n times back-to-back, discard first, keep rest.

reports min/median/stddev per (engine, file, mode) and a compact
comparison table at the end. produces results.json for later analysis.
"""
import argparse
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
X8R = ROOT.parent / "build" / "x8r"
DUNCACHE = Path("/tmp/duncache/duncache")

try:
    import tiktoken
except ImportError:
    print("need: pip install tiktoken", file=sys.stderr)
    sys.exit(2)

MODEL = os.environ.get("X8R_BENCH_MODEL", "cl100k")
if MODEL not in ("cl100k", "o200k"):
    raise SystemExit(f"X8R_BENCH_MODEL must be cl100k or o200k, got {MODEL!r}")
ENC = tiktoken.get_encoding(f"{MODEL}_base")
VOCAB = ROOT.parent / "vocab" / f"{MODEL}.bin"
print(f"benchmarking {MODEL} vs tiktoken {MODEL}_base", file=sys.stderr)

def evict(path: Path):
    subprocess.run([str(DUNCACHE), "-q", "--no-sync", str(path)], check=True)

def time_x8r(path: Path) -> tuple[float, int]:
    t0 = time.perf_counter_ns()
    env = dict(os.environ, X8R_VOCAB=str(VOCAB))
    r = subprocess.run([str(X8R), "--count", str(path)], capture_output=True, check=True, env=env)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, int(r.stdout.strip())

def time_tiktoken(path: Path) -> tuple[float, int]:
    # include file read in the measurement to be fair to x8r (it reads too)
    t0 = time.perf_counter_ns()
    data = path.read_bytes()
    toks = ENC.encode_ordinary(data.decode("utf-8", errors="strict"))
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, len(toks)

ENGINES = {
    "x8r": time_x8r,
    "tiktoken": time_tiktoken,
}

def stats(samples: list[float]) -> dict:
    if not samples:
        return {}
    return {
        "n": len(samples),
        "min": min(samples),
        "median": st.median(samples),
        "mean": st.mean(samples),
        "stddev": st.stdev(samples) if len(samples) > 1 else 0.0,
        "max": max(samples),
    }

def run_cold(fn, path: Path, iters: int) -> list[float]:
    out = []
    for _ in range(iters):
        evict(path)
        ms, _ = fn(path)
        out.append(ms)
    return out

def run_warm(fn, path: Path, iters: int) -> list[float]:
    fn(path)  # prime
    out = []
    for _ in range(iters):
        ms, _ = fn(path)
        out.append(ms)
    return out

def verify_match(path: Path) -> tuple[int, int]:
    _, a = time_x8r(path)
    _, b = time_tiktoken(path)
    return a, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--cold-iters", type=int, default=5)
    ap.add_argument("--json-out", default=str(ROOT / "results.json"))
    ap.add_argument("--only", help="glob filter on corpus filenames")
    args = ap.parse_args()

    if not X8R.exists():
        print(f"missing {X8R}; run 'make' first", file=sys.stderr); sys.exit(2)
    if not DUNCACHE.exists():
        print(f"missing {DUNCACHE}; build /tmp/duncache first", file=sys.stderr); sys.exit(2)

    files = sorted(CORPUS.glob("*.txt"))
    if args.only:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(f.name, args.only)]
    if not files:
        print("no corpus files; run gen_corpus.py", file=sys.stderr); sys.exit(2)

    # correctness check first: token counts must agree
    print("verifying token counts match tiktoken...")
    for f in files:
        a, b = verify_match(f)
        status = "ok" if a == b else f"MISMATCH x8r={a} tik={b}"
        print(f"  {f.name}: {a} tokens [{status}]")
        if a != b:
            print("aborting: counts diverge", file=sys.stderr); sys.exit(1)

    results = {}
    print(f"\nbenchmarking (cold_iters={args.cold_iters}, warm_iters={args.iters})...")
    for f in files:
        size = f.stat().st_size
        results[f.name] = {"size": size}
        for eng, fn in ENGINES.items():
            cold = run_cold(fn, f, args.cold_iters)
            warm = run_warm(fn, f, args.iters)
            results[f.name][eng] = {
                "cold": stats(cold),
                "warm": stats(warm),
            }
            print(f"  {f.name:32s} {eng:9s} cold_min={min(cold):7.2f}ms warm_min={min(warm):7.2f}ms")

    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")

    # summary table
    print("\n" + "=" * 90)
    print(f"{'file':32s} {'size':>10s}  {'x8r_cold':>10s} {'tik_cold':>10s} {'spd_cold':>8s}  {'x8r_warm':>10s} {'tik_warm':>10s} {'spd_warm':>8s}")
    print("-" * 90)
    for name, r in results.items():
        size = r["size"]
        xc = r["x8r"]["cold"]["min"]; tc = r["tiktoken"]["cold"]["min"]
        xw = r["x8r"]["warm"]["min"]; tw = r["tiktoken"]["warm"]["min"]
        print(f"{name:32s} {size:>10d}  {xc:>9.2f}ms {tc:>9.2f}ms {tc/xc:>7.2f}x  {xw:>9.2f}ms {tw:>9.2f}ms {tw/xw:>7.2f}x")

if __name__ == "__main__":
    main()
