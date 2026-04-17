#!/usr/bin/env python3
"""
generate reference token counts for golden test inputs using tiktoken.

for every file in tests/golden/in/*, writes tests/golden/expected/<name>.count
containing a single integer (cl100k token count).
"""
import sys
from pathlib import Path


def main():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    root = Path(__file__).resolve().parents[1]
    in_dir = root / "tests" / "golden" / "in"
    out_dir = root / "tests" / "golden" / "expected"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.iterdir()) if in_dir.exists() else []
    if not files:
        print(f"no files in {in_dir}", file=sys.stderr)
        return 1
    for f in files:
        data = f.read_bytes()
        # use encode_ordinary to skip special-token handling
        n = len(enc.encode_ordinary(data.decode("utf-8", errors="replace")))
        (out_dir / (f.name + ".count")).write_text(f"{n}\n")
        print(f"{f.name}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
