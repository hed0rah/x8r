#!/usr/bin/env bash
# bench/perf_vocab.sh
#
# Measure cache hierarchy behaviour of x8r_vocab_lookup via perf stat.
#
# Usage:  bench/perf_vocab.sh <corpus-file> [iterations]
#
# Runs the binary multiple times under perf stat with available cache
# events.  The fine-grained Intel events (mem_load_retired.*, l2_rqsts.*,
# l1d_pend_miss.fb_full) are requested first; if unavailable the script
# falls back to generic events (L1-dcache, LLC, cache-misses).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <corpus-file> [iterations]"
    echo "  e.g. $0 bench/corpus/cjk_dense.txt 7"
    exit 1
fi

CORPUS="$1"
ITER=${2:-7}

if [ ! -f "$CORPUS" ]; then
    echo "Error: corpus file not found: $CORPUS"
    exit 1
fi

BINARY="./build/x8r"
if [ ! -x "$BINARY" ]; then
    echo "Error: binary not found: $BINARY (run make first)"
    exit 1
fi

# --- permission check ---
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
if [ "$PARANOID" -ge 2 ]; then
    echo "================================================================"
    echo "ERROR: perf_event_paranoid = $PARANOID (>= 2), cannot measure."
    echo "To enable:"
    echo "  sudo sysctl kernel.perf_event_paranoid=1"
    echo "Or (persistent):"
    echo "  echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf"
    echo "================================================================"
    exit 1
fi

# --- event selection: try fine-grained Intel events, fall back to generics ---

INTEL_EVENTS="\
mem_load_retired.l1_hit,\
mem_load_retired.l1_miss,\
mem_load_retired.l2_hit,\
mem_load_retired.l3_hit,\
mem_load_retired.l3_miss,\
l2_rqsts.all_demand_data_rd,\
l2_rqsts.pf_hit,\
l2_rqsts.pf_miss,\
l1d_pend_miss.fb_full,\
cycles,\
instructions"

GENERIC_EVENTS="\
L1-dcache-loads,\
L1-dcache-load-misses,\
LLC-loads,\
LLC-load-misses,\
cache-misses,\
cycles,\
instructions"

USE_INTEL=false
if perf stat -e "$INTEL_EVENTS" true > /dev/null 2>&1; then
    USE_INTEL=true
    EVENTS="$INTEL_EVENTS"
else
    echo "Note: Intel-specific events not available in this environment."
    echo "Falling back to generic cache events."
    EVENTS="$GENERIC_EVENTS"
fi

echo "================================================================"
echo "  x8r vocab cache perf measurement"
echo "  file:      $CORPUS"
echo "  iterations: $ITER"
echo "  events:    Intel=$USE_INTEL"
echo "================================================================"

# Warm-up run
"$BINARY" --count "$CORPUS" > /dev/null 2>&1 || true

# Collect raw outputs per iteration
ALL_OUTPUT=""

for i in $(seq 1 "$ITER"); do
    echo ""
    echo "--- iteration $i / $ITER ---"
    perf stat -e "$EVENTS" "$BINARY" --count "$CORPUS" 2>&1
done

echo ""
echo "===== Summary (last iteration) ====="
echo ""

if $USE_INTEL; then
    cat <<'SUMM'
With Intel events, compute:
  L1 hit rate   = l1_hit / (l1_hit + l1_miss)
  L2 hit rate   = l2_hit / (l1_miss)
  L3 hit rate   = l3_hit / (l1_miss - l2_hit)
  DRAM rate     = l3_miss / (l1_miss - l2_hit - l3_hit)
  HW prefetch L2 hit = pf_hit / (pf_hit + pf_miss)
  FB full ratio = fb_full / (l1_miss)
  CPI          = cycles / instructions
SUMM
else
    cat <<'SUMM'
With generic events, compute:
  L1 miss rate  = L1-dcache-load-misses / L1-dcache-loads
  LLC hit rate  = (LLC-loads - LLC-load-misses) / LLC-loads
  LLC miss rate = LLC-load-misses / LLC-loads
  cache-miss rate = cache-misses / L1-dcache-loads
  CPI          = cycles / instructions
SUMM
fi