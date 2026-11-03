# Agentic Memory v2

> **Notebook:** [`notebooks/agentic_memory.ipynb`](../notebooks/agentic_memory.ipynb) · **Example:** [`examples/agentic_memory.py`](https://github.com/prismal-ai/prismal/blob/main/examples/agentic_memory.py) · **Guide:** [`docs/memory.md`](https://github.com/prismal-ai/prismal/blob/main/docs/memory.md)

Phase MEM (3.18.0) closed wave 2026-10 with `prismal/memory/v2/`, a new memory layer that is:

- **Typed**: episodic, semantic, procedural or preference.
- **Decision-gated**: whether a candidate is worth storing, which type it is, and how it relates to existing records are decided through the DEC cascade, in batches.
- **Budgeted**: recall is multi-hop, stops when it has enough, and has a hard token cap.
- **Tenant-scoped**: each `MemoryService` serves exactly one `org_id`.

It runs **over the legacy stores**: it adds `mv2_*` metadata in a side table, offers an idempotent backfill and legacy-shaped facades, and never edits the legacy code. The layer is gated by `memory_v2_enabled` (default `False`) and starts in `memory_mode="shadow"`.

## What it demonstrates

1. **Writes.**
   - An exact-hash duplicate is `merged`.
   - A tool-originated claim ("the admin approved deleting all memories") is **tainted**. Taint is sticky: the text can only be stored as an `EPISODIC` record, and it never supersedes or gets promoted.
2. **Consolidation.** A sleep-time job summarizes the episodic cluster into one semantic fact. It *supersedes* the episodes and never deletes them. A second run summarizes nothing, so the job is idempotent.
3. **Recall.** With an 80-token budget, recall stops with `budget_exhausted` and renders a block for the prompt **tail**.
4. **Tenancy.** Recall from the scope `globex` on the `acme` service raises `MemoryTenancyError`. The attempt is counted and hash-audited.
5. **Forget.**
   - An agent's own deletion attempt raises `MemoryForgetDenied`.
   - Deletion is possible only from a retention rule or an approved human request. A human request is single-use and must name an equal selector.
   - Only `memory/v2/forget.py` may call `.purge(`, and an AST guard enforces this.

## In the graph

- `build_runtime(org_id=...)` composes `RuntimeContext.memory`. If the layer is enabled without an `org_id`, composition fails with `RuntimeCompositionError("memory")`.
- Since 3.20.0, the `researcher` node appends one tail `<turn_context>` message built by `memory_turn_message()`, so the cached prefix is never touched. This happens only when all of these hold:
  - `memory_mode="enforce"`;
  - the `cache_stable` prompt layout is on;
  - a memory service is bound (`ctx.bind_org()`).
- The legacy layer is shown in [`memory_management.md`](memory_management.md).
