#!/usr/bin/env bash
# runs every file in tests/golden/in through x8r --count and compares to
# tests/golden/expected/<name>.count. exits non-zero on any mismatch.
set -u
cd "$(dirname "$0")/.."

bin=${X8R_BIN:-./build/x8r}
vocab=${X8R_VOCAB:-./vocab/cl100k.bin}

if [ ! -x "$bin" ]; then echo "missing binary: $bin (run make)" >&2; exit 2; fi
if [ ! -f "$vocab" ]; then echo "missing vocab: $vocab (run scripts/dump_cl100k.py)" >&2; exit 2; fi

fail=0
pass=0
for f in tests/golden/in/*; do
    name=$(basename "$f")
    exp_file="tests/golden/expected/$name.count"
    if [ ! -f "$exp_file" ]; then
        echo "SKIP $name (no expected file)"
        continue
    fi
    expected=$(cat "$exp_file")
    actual=$(X8R_VOCAB="$vocab" "$bin" --count "$f")
    if [ "$expected" = "$actual" ]; then
        pass=$((pass+1))
        echo "OK   $name ($actual)"
    else
        fail=$((fail+1))
        echo "FAIL $name expected=$expected got=$actual"
    fi
done

echo "---"
echo "$pass passed, $fail failed"
exit $fail
