# Muse-Glimmer-30B: agentic + vision validation (tq4 vs tq3)

Date: 2026-08-11 · Machine: 64 GB M4 Max · Engine: turboquant-mlx 0.21.0, re-verified on 0.21.1
Client: opencode 1.17.13 · Server: `turboquant-serve-vlm` (mlx-vlm 0.6.12)

First agentic validation of a **dense** TurboQuant model, and of
`turboquant-serve-vlm`. Both builds pass 3/3, and quantizing the vision tower
costs nothing measurable against the build that leaves it in bf16.

## 1. Agentic coding — OpenCode, n=3 per build

```bash
turboquant-serve-vlm --model <path> --port 8080 --reasoning-strength low
```

- Task: identical to the 35B runs (`setup_task_repo.sh`) — planted
  `len(values) + 1` off-by-one in `average()`, failing pytest. The prompt gives
  the exact venv command and asks to run → find → fix → re-run → explain.
- Sampling: **temp 1.0 / top_p 0.95** (Meta's recommendation for this model),
  set via opencode's `agent.build`. Opencode sends no temperature of its own and
  mlx-vlm's server defaults to 0.0 (greedy), which is the configuration that
  produced the 35B's 204x perseveration loop — so this is load-bearing.
- Skills disabled (`"tools": {"skill*": false}`) for a clean 7.3K system prompt.
- Fresh task repo per run; server restarted per build, one model resident.

| build | run | secs | requests | actions | fix | tests intact |
|---|---|---|---|---|---|---|
| `tq4-g64` | 1 | 547 | 7 | 4 | ✅ | ✅ |
| `tq4-g64` | 2 | 658 | 7 | 4 | ✅ | ✅ |
| `tq4-g64` | 3 | 658 | 7 | 4 | ✅ | ✅ |
| `tq3-g64` | 1 | 795 | 8 | 5 | ✅ | ✅ |
| `tq3-g64` | 2 | 728 | 8 | 5 | ✅ | ✅ |
| `tq3-g64` | 3 | 640 | 7 | 4 | ✅ | ✅ |

**tq4 3/3 · tq3 3/3.** Trajectories:

```
tq4:  pytest → Read . → Read stats.py → edit → pytest → 3 passed
tq3:  ls -la → pytest → Read stats.py → Read test_stats.py → edit → pytest → 3 passed
```

Every run executed the exact venv command it was given, first try — which the
35B ternary never did across 4 configs (see RESULTS.md). Action counts stayed in
a 4–5 band with no run drifting; the perseveration failure mode would show up as
a run ballooning to 20+ actions.

**A pass requires all three of:** pytest reports `3 passed`, the fix is
specifically `sum(values) / len(values)`, and `test_stats.py` is byte-identical
to a freshly generated reference. The last one matters — "3 passed" alone cannot
distinguish a real fix from a model that edited the tests until they agreed with
the bug. All 6 runs were byte-identical.

### The missing closing explanation — found and fixed in 0.21.1

The prompt ends with "Tell me what the bug was." **All 8 runs on 0.21.0 end at
`3 passed` with zero lines of prose after it.** The engineering was right every
time; the summary was absent, consistently.

It was **our bug, in the server** — see §3 for the root cause and the fix. After
0.21.1, one run per build:

| build | secs | requests | actions | fix | tests intact | prose lines |
|---|---|---|---|---|---|---|
| `tq4-g64` | 589 | 7 | 4 | ✅ | ✅ | **6** |
| `tq3-g64` | 772 | 8 | 5 | ✅ | ✅ | **6** |

> The bug was in `stats.py:8`. The average function divided by `len(values) + 1`
> instead of `len(values)`.

Action counts, the fix and the test-file md5 are all unchanged, so tool calling
is unaffected by the fix.

## 2. Vision — quantized tower vs bf16 control

The size advantage over `mlx-community/Muse-Glimmer-30B-4bit` exists largely
because TurboQuant quantizes `embed_tokens` **and the entire 50-layer ViT-G/14
vision tower** (~1.8B params) that the affine build leaves in bf16. So vision is
exactly where our compression could silently break — and it had never been
tested on the quantized builds.

Four synthetic images with exact ground truth (greedy decode, so re-runnable).
The mlx-community build is included as a **control**: a case both miss is a model
limitation, a case only ours misses is our bug.

| case | ground truth | tq4 | tq3 | mlx4 control (bf16 vision) |
|---|---|---|---|---|
| OCR | `VOLTAGE 47` | ✅ 13 s | ✅ 12 s | ✅ 547 s ⁽¹⁾ |
| count red circles (blue-square distractors) | `3` | ✅ 11 s | ✅ 11 s | ✅ 7 s |
| tallest bar | `C` | ✅ 11 s | ✅ 11 s | ✅ 7 s |
| top-left shape | `triangle` | ✅ 11 s | ✅ 13 s | ✅ 7 s |
| | | **4/4** | **4/4** | **4/4** |

⁽¹⁾ cold load of a 19.88 GiB model, not inference — the following three cases at
7 s confirm it. Our builds were already page-cached from the agentic runs.

**Quantizing the vision tower cost nothing measurable here.** Two limits on that
claim, stated plainly:

- 4/4 vs 4/4 means **no regression detected**, not identical vision quality.
  These are clean synthetic images with unambiguous answers; a harder battery
  (cluttered photos, small text, fine-grained counting) could still separate
  them. This is a soundness check, not a vision benchmark.
- The control is **faster per case once warm** (7 s vs 11–13 s), which is the
  codebook-vs-affine decode gap already documented for this model (2.4–2.8×).

## 3. Reasoning level: `low` vs `medium`

`low` is what the model cards recommend for agents. `medium` was run to ask two
things: does more deliberation help on a roomier machine (24 GB+, where the
token budget is not the constraint it is on a 16 GB mini), and does it bring
back the missing explanation?

| build | level | secs | requests | actions | pass | prose after final test |
|---|---|---|---|---|---|---|
| tq4 | low (n=3) | 547 / 658 / 658 | 7 | 4 | 3/3 | 0 |
| tq4 | medium | 727 | 8 | 5 | ✅ | 0 |
| tq3 | low (n=3) | 640 / 728 / 795 | 7–8 | 4–5 | 3/3 | 0 |
| tq3 | medium | 608 | 7 | 4 | ✅ | 0 |

**Medium passes, and buys nothing measurable.** tq4 costs +70 s and one extra
action for the same correct fix; tq3 lands at 608 s, inside the `low` spread
(640–795 s), so the wall-clock difference is within run-to-run noise rather than
a real slowdown. No behavioural difference worth a recommendation change:
**stay on `low`**. `high` was not run — on this evidence it would only be slower
for the same outcome.

### The missing explanation was ours, and two hypotheses were wrong

**Hypothesis 1 — the reasoning level trims the summary. Falsified.** `medium`
produces `prose_lines = 0` on both builds too. The raw transcript was checked
directly, so it was never a parsing artifact.

**Hypothesis 2 — the model never emits the `to=user` routing header, so the
thinking splitter never closes. Also wrong.** The header is emitted. The
content is discarded *after* the splitter, further down the pipe.

**Actual root cause.** ATEM declares
`tool_call_start = "to=self<|message|>"` — byte-identical to the channel router
Muse Glimmer emits at the start of every turn. While streaming, mlx-vlm drops
every content delta from the moment `tool_call_start` appears in the output, and
`in_tool_call` **has no release path**: once latched, it stays latched for the
rest of the generation. So on any request that declared tools, the latch closed
on the first reasoning token and nothing the model said afterwards reached the
client.

Reproducible with no model loaded:

```python
from mlx_vlm.server.responses_state import suppress_tool_call_content as S
R = "to=self<|message|>"
S(R + "thinking", False, R, "the answer")   # -> (True, None)   the answer, dropped
S("...", False, None, "the answer")         # -> (False, 'the answer')  no tools: fine
```

That asymmetry is why it read as a model problem: a plain curl answered
correctly (no tools declared, so no trigger), tool calls kept working (they are
parsed from the *full output*, not the stream), and only prose went missing.

**Fix (0.21.1):** `turboquant-serve-vlm` remaps the suppression trigger to
`<atem:function_calls>`, the tag that genuinely opens a call — and the string
mlx-vlm itself uses to detect the ATEM format. Tool-call parsing is untouched,
since `process_tool_calls` reads the marker from the parser. Verified in §1.

**The transferable lesson:** when a model's channel protocol and its tool
protocol share a marker, every consumer that keys off that marker becomes
ambiguous. Check what the inferred tool parser declares against what the model
emits on an *ordinary* turn, not just a tool turn.

## 4. Two serving gotchas specific to `turboquant-serve-vlm`

Neither exists with `turboquant-serve`, and both cost real time here.

1. **The model id must be the exact path the server preloaded.** mlx-vlm's
   server routes by `request.model` and will fetch+load *any* other id from the
   Hub. `default_model` (the `turboquant-serve` convention) 404s, and sending the
   upstream repo id `meta-models/Muse-Glimmer-30B` **started loading the 59.6 GB
   bf16 original over the top of the quantized one** — caught at 2.3 GB RSS and
   killed. On a 64 GB box that is a wedge risk.
2. **`max_tokens` must clear the reasoning budget.** At `max_tokens: 20` the
   reply came back `content: None` — reasoning consumed the whole allowance
   before any answer token. Fine at 150; opencode allows 8192. A client with a
   tight cap sees empty replies and blames the model.

## Reproduce

```bash
bash scripts/opencode_smoke/setup_task_repo.sh /tmp/task
turboquant-serve-vlm --model manjunathshiva/Muse-Glimmer-30B-tq4-g64 \
    --port 8080 --reasoning-strength low
# opencode.json: provider baseURL http://127.0.0.1:8080/v1, model id = the
# exact --model path, agent.build.temperature 1.0 / top_p 0.95
cd /tmp/task && opencode run "<the prompt above>"
```

Vision battery: `make_vision_tests.py` + `run_vision_tests.py` (scratchpad).
