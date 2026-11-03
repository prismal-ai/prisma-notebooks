# Observation Masking and `read_artifact`

> **Notebook:** [`notebooks/observation_masking.ipynb`](../notebooks/observation_masking.ipynb) · **Source:** [`prismal/agents/context_compaction.py`](https://github.com/prismal-ai/prismal/blob/main/prismal/agents/context_compaction.py), [`prismal/agents/artifacts.py`](https://github.com/prismal-ai/prismal/blob/main/prismal/agents/artifacts.py) · **Guide:** [`docs/token-efficiency.md`](https://github.com/prismal-ai/prismal/blob/main/docs/token-efficiency.md)

In a long ReAct loop, most of the context is old tool observations that the model has already used. `CompactionStrategy.MASK_OBSERVATIONS` (TKE4) moves those payloads into an `ArtifactStorePort` and replaces each one *in place* with a short stub:

```
[observation elided: tool=search_orders artifact=<id> tokens≈2482; use read_artifact]
```

The stub keeps the original message `id`, `tool_call_id`, `name` and `status`, so tool-call pairing stays valid. `to_state_update()` emits same-id messages, and LangGraph's `add_messages` reducer swaps them in. No message is dropped. In the notebook, a 15,066-character history becomes 244 characters.

## What changed in 3.20.0

- **Wired into the loop.** `react_loop` had accepted a `context_compactor` argument since LH1-06, but no agent passed one. All nine agent nodes do now, and an AST guard keeps the call paired with `hardening_react_kwargs`.
- **One shared store.** The compactor and the tool-output budget use one artifact store per process and backend (`shared_artifact_store`). Before, an artifact written by one could not be read back through the other.
- **Bounded store.** The store holds at most 1,000 artifacts per tenant (a tenant over the limit evicts only its own oldest) and 10,000 overall. Expired artifacts are purged on `put`, at most once a minute.
- **`read_artifact`.** The tool is bound to the loop's own `(tenant, run_id)`. Reads are windowed to 8,000 characters and end with a `[more: call read_artifact with start=…]` pointer. Reads are not re-budgeted, and they go through the same sanitizer and Phase H screen as the original tool output.
- **Stub wording.** The stub now ends with "use read_artifact". The TKE9-04 live run showed why: when the stub did not name the tool, models re-read elided data 0 times out of 3.

## Scope isolation

An unknown id, or a known id read from another run or another tenant, returns `"artifact not found"`. It never raises into the agent loop. Each anonymous run gets an unguessable `anon-<hex>` scope, so two such runs can never read each other's artifacts.
