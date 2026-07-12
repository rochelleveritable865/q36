#!/usr/bin/env bash
# This file is part of q36, a dedicated CUDA inference engine for
# Qwen3.6-35B-A3B.
#
# Copyright (C) 2026 Ambud Sharma
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with this program.  If not, see
# <https://www.gnu.org/licenses/>.

# Long-context retrieval regression test: hide a passphrase at several depths
# inside filler, ask for it back, check the greedy answer.  Run on the GPU
# host from the repository root (expects ./q36 built).
#
#   ./scripts/needle_sweep.sh [ctx_tokens ...]     default: 16384 32768 90112
#
# Exercises both KV modes at each size.  A FAIL means long-context attention
# or the SSM state is broken -- treat like a Paris/391 failure.
set -u
BIN=${BIN:-./q36}
SIZES=(${@:-16384 32768 90112})
NEEDLE="turquoise-elephant-942"
FILLER="The quick brown fox jumps over the lazy dog near the riverbank while autumn leaves drift slowly across the quiet meadow under a pale morning sky. "

pass=0; fail=0
for n in "${SIZES[@]}"; do
    # ~33 tokens per filler sentence; needle at ~25% depth
    reps=$(( n / 33 ))
    at=$(( reps / 4 ))
    prompt=$(python3 - "$reps" "$at" "$NEEDLE" "$FILLER" <<'EOF'
import sys
reps, at, needle, filler = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
parts = [filler] * reps
parts[at] = f"The secret passphrase is {needle}. Remember it. "
parts.append("\nWhat is the secret passphrase? Answer with just the passphrase.")
sys.stdout.write("".join(parts))
EOF
)
    for kv in "" "--kv-quant"; do
        out=$(printf '%s\n' "$prompt" | $BIN --ctx $((n + 2048)) -n 2048 $kv 2>/dev/null | tr -cd '\11\12\15\40-\176')
        if grep -q "$NEEDLE" <<<"$out"; then
            echo "PASS  ctx~$n  ${kv:-fp16-kv}"
            pass=$((pass+1))
        else
            echo "FAIL  ctx~$n  ${kv:-fp16-kv}"
            fail=$((fail+1))
        fi
    done
done
echo "----"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
