# Multi-session prefix cache (NVIDIA server)

`dsx-server` keeps one physical sequence: one engine thread, one `Real`, one `GpuModel`, one request at a time. On
top of that, `--cache-sessions N > 1` retains several independent conversation lineages so sequential coding-agent
sessions can switch back to long histories without prefilling them from token 0.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--cache-sessions N` | `1` | lineages kept, including the active one. `1` is the backward-compatible fast path: no host parking, no arena, no session switches. |
| `--session-cache-gb` | `8` | hard budget for the host cache of inactive lineages, decimal GB (committed arena bytes and the heap-side cached keys, saved logits and Engram histories). Ignored with `--cache-sessions 1`. |
| `--prefix-anchor-tokens` | `16384` | sparse anchor interval in prompt tokens; `0` disables anchors. Anchors only matter in multi-session mode. |

Cache pressure only costs latency: requests still succeed and fall back to a cold prefill when a lineage is not
retained.

## Usage

```sh
dsx-server --model ... --cache-sessions 4 --session-cache-gb 8 --prefix-anchor-tokens 16384
```

Two scenarios are covered:

1. `A -> B -> A`: A is parked when B runs and restored when A returns; only the new tail is prefilled.
2. `A` and `B` share a large system/tools prefix: B rewinds to a sparse anchor of A and prefills only the divergent
   tail.

The extra flags pay off only when requests keep switching between long histories; a single interactive conversation
is served by the default `--cache-sessions 1`.

## Observability

Requests emit one `event=session_cache` line with the selected session/source, action, `raw_lcp`/`usable_start`,
replayed tokens, park/restore bytes and milliseconds, and the arena payload/committed/budget. Parking, anchor
capture, eviction and admission rejection have their own structured events (for example
`session_cache_admission result=rejected reason=memory-budget`).

The OpenAI-compatible usage field does not report internal cache hits; only the server log does.

## Limitations (v1)

- sequential requests only; no continuous batching, paged KV, radix tree, disk or remote cache;
- eviction is whole-session; anchors are not evicted individually, and anchor positions are chunk-aligned;
- restoring an inactive source copies its full backing state even when only an anchor is needed for a clone;
- no TTL: the session count, the byte budget and LRU are the only bounds;
- only a local smoke test has been run so far, not a controlled benchmark.
