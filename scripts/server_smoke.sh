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

# q36_server HTTP smoke test: exercises both wire protocols end to end.
# Launches two server instances in sequence (default flags, then
# --hide-think --temp 0) and checks the response shapes an agent client
# depends on: Anthropic thinking/text/tool_use content blocks, SSE event
# grammar, think suppression, tool-call parsing, greedy default temp.
# Run on the GPU host from the repository root (expects ./q36_server built).
#
#   MODEL=/path/to.gguf ./scripts/server_smoke.sh [port]     default: 8399
#
# A FAIL means the server wire format regressed -- agent harnesses (Claude
# Code etc.) will misrender or drop content.  ~1 min wall on a warm cache.
set -u
BIN=${BIN:-./q36_server}
MODEL=${MODEL:?set MODEL=/path/to/model.gguf}
PORT=${1:-8399}
URL="http://localhost:$PORT"
LOG=$(mktemp /tmp/q36_smoke.XXXXXX.log)
pass=0; fail=0; srv=

check(){ # check <name> <ok:0|1>
    if [ "$2" -eq 0 ]; then echo "PASS  $1"; pass=$((pass+1));
    else echo "FAIL  $1"; fail=$((fail+1)); fi
}
msg(){ curl -s -m 180 "$URL/v1/messages" -H 'content-type: application/json' -d "$1"; }
oai(){ curl -s -m 180 "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$1"; }

start(){ # start [extra flags...]
    : >"$LOG"
    "$BIN" -m "$MODEL" --port "$PORT" "$@" >>"$LOG" 2>&1 & srv=$!
    for _ in $(seq 1 90); do grep -q 'engine ready' "$LOG" && return 0; sleep 2; done
    echo "FATAL server did not come up; log tail:"; tail -5 "$LOG"; exit 1
}
stop(){ [ -n "$srv" ] && kill -TERM "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; srv=; }
trap 'stop; rm -f "$LOG"' EXIT

# ---- instance 1: default flags -------------------------------------------
start

# Anthropic non-stream, greedy: text present, and any thinking block must
# have real content (whitespace-only think regions emit no block)
msg '{"model":"q36","max_tokens":300,"temperature":0,
     "messages":[{"role":"user","content":"What is 17*23? Answer briefly."}]}' \
| python3 -c '
import json,sys
m=json.load(sys.stdin); c=m["content"]
assert any(b["type"]=="text" and "391" in b["text"] for b in c), c
assert all(b["thinking"].strip() for b in c if b["type"]=="thinking"), c
assert m["stop_reason"]=="end_turn" and m["usage"]["output_tokens"]>0'
check "anthropic non-stream: text + no empty thinking block" $?

# Anthropic SSE grammar: block starts/stops pair up with ascending indices,
# thinking (if any) is non-blank, no <think> tag leaks.  Sampled generation
# can burn the whole budget thinking -- then grammar must still hold, but
# the 391 answer is only required on a normal end_turn.
msg '{"model":"q36","max_tokens":500,"stream":true,
     "messages":[{"role":"user","content":"What is 17*23? Answer briefly."}]}' \
| python3 -c '
import json,sys
evs=[json.loads(l[6:]) for l in sys.stdin if l.startswith("data: ")]
assert evs[0]["type"]=="message_start" and evs[-1]["type"]=="message_stop", evs[:1]
open_,order,think,text,stop={},[],"","",None
for e in evs:
    t=e["type"]
    if t=="content_block_start":
        i=e["index"]; assert i not in open_ and (not order or i>order[-1]), e
        open_[i]=e["content_block"]["type"]; order.append(i)
    if t=="content_block_delta":
        d=e["delta"]; i=e["index"]; assert i in open_, e
        if d["type"]=="thinking_delta": think+=d["thinking"]
        if d["type"]=="text_delta":     text+=d["text"]
    if t=="content_block_stop": del open_[e["index"]]
    if t=="message_delta": stop=e["delta"]["stop_reason"]
assert not open_ and "<think" not in text, (open_,text)
assert think=="" or think.strip(), repr(think)
assert "391" in text or stop=="max_tokens", (stop,text)
' ; check "anthropic stream: SSE grammar + no tag leak" $?

# tool call: stop_reason tool_use, typed input parsed
msg '{"model":"q36","max_tokens":400,
     "tools":[{"name":"get_weather","description":"Get current weather for a city",
       "input_schema":{"type":"object","properties":{"city":{"type":"string"}},
       "required":["city"]}}],
     "messages":[{"role":"user","content":"Use the weather tool to check Paris."}]}' \
| python3 -c '
import json,sys
m=json.load(sys.stdin)
tu=[b for b in m["content"] if b["type"]=="tool_use"]
assert m["stop_reason"]=="tool_use" and tu and tu[0]["name"]=="get_weather", m
assert "aris" in json.dumps(tu[0]["input"]), tu'
check "anthropic tool call: tool_use block + typed input" $?

# OpenAI wire, flags off: raw <think> text is preserved (back-compat)
oai '{"model":"q36","max_tokens":300,"temperature":0,
     "messages":[{"role":"user","content":"What is 17*23? Answer briefly."}]}' \
| python3 -c '
import json,sys
c=json.load(sys.stdin)["choices"][0]
assert c["finish_reason"]=="stop" and "391" in c["message"]["content"], c'
check "openai non-stream: flags-off back-compat" $?

# unterminated think (max_tokens mid-region) must still close every block
msg '{"model":"q36","max_tokens":15,"stream":true,
     "messages":[{"role":"user","content":"Explain quantum entanglement in detail."}]}' \
| python3 -c '
import json,sys
evs=[json.loads(l[6:]) for l in sys.stdin if l.startswith("data: ")]
starts=sum(e["type"]=="content_block_start" for e in evs)
stops =sum(e["type"]=="content_block_stop"  for e in evs)
d=[e for e in evs if e["type"]=="message_delta"]
assert starts==stops and d and d[0]["delta"]["stop_reason"]=="max_tokens", evs[-3:]'
check "anthropic stream: max_tokens mid-think closes blocks" $?

stop

# ---- instance 2: --hide-think --temp 0 ------------------------------------
start --hide-think --temp 0
grep -q '(greedy)' "$LOG"; check "--temp 0: startup reports greedy default" $?

msg '{"model":"q36","max_tokens":300,
     "messages":[{"role":"user","content":"What is 17*23? Answer briefly."}]}' \
| python3 -c '
import json,sys
c=json.load(sys.stdin)["content"]
assert not [b for b in c if b["type"]=="thinking"], c
assert any(b["type"]=="text" and "391" in b["text"] for b in c), c'
check "--hide-think anthropic: no thinking block" $?

oai '{"model":"q36","max_tokens":300,
     "messages":[{"role":"user","content":"What is 17*23? Answer briefly."}]}' \
| python3 -c '
import json,sys
c=json.load(sys.stdin)["choices"][0]["message"]["content"]
assert "<think" not in c and "391" in c, repr(c)'
check "--hide-think openai: think region suppressed" $?

stop
echo "----"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
