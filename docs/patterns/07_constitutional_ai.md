# Constitutional AI

> **Notebook:** [`notebooks/patterns/07_constitutional_ai.ipynb`](../../notebooks/patterns/07_constitutional_ai.ipynb) · **Example:** [`examples/patterns/07_constitutional_ai.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/07_constitutional_ai.py)

Constitutional AI is a principle-driven self-revision pattern: instead of trusting a single generation, the model's output is checked against an explicit list of written principles (a "constitution"), and every detected violation triggers an LLM-powered rewrite. The check → revise cycle repeats until the principle is satisfied or a revision cap is hit, and every change is recorded for audit. This gives you a policy layer you can read, version, and extend — far more transparent than a black-box safety classifier.

Reach for this pattern when agent output must comply with rules you can articulate — no harmful content, no PII leakage, no unhedged false claims, house style, regulatory language — and when you need an audit trail proving *what* was flagged and *how* it was fixed. The notebook exercises the filter on a small evaluation set inspired by AdvBench (adversarial prompts) and SafeNLP: five texts spanning safe security advice, a support reply leaking customer PII, a wildly inaccurate computing-history paragraph, discriminatory financial advice, and a blog post mixing PII with absolute claims.

## What it demonstrates

- Running `ConstitutionalFilter.apply()` with the three built-in `DEFAULT_PRINCIPLES` — P001 `no_harmful_content` (critical), P002 `factual_accuracy` (high), P003 `no_pii_exposure` (high).
- Defining **custom principles** (`ConstitutionalPrinciple` dataclass) — an anti-discrimination rule and an "epistemic humility" rule — and combining them with the defaults.
- Interpreting a `ConstitutionalResult`: `final_output`, the ordered `revisions` list, `all_principles_satisfied`, and the `max_revisions_reached` escape hatch.
- The bounded revise/check loop: up to `max_revisions` attempts per principle, so a stubborn violation can never spin forever.
- A statistical pass over the whole dataset: violations per principle, fully-safe texts, and total revisions applied.

## How it works

### Principles as data, not code

Each rule is a plain dataclass carrying two prompts — one to *detect* a violation, one to *fix* it — plus a severity. The critique prompt instructs the model to answer with a strict `VIOLATION:`/`OK:` prefix, which the filter parses conservatively (unparseable replies count as no violation, to avoid over-blocking benign text):

```python
ConstitutionalPrinciple(
    id="P005",
    name="epistemic_humility",
    description="The output must not present uncertain claims as absolute certainties.",
    critique_prompt=(
        "Does the text present uncertain or speculative claims as absolute, "
        "irrefutable facts? Respond 'YES: [problematic statements]' or 'NO'."
    ),
    revision_prompt=(
        "Rewrite the text adding appropriate epistemic qualifiers "
        "('may', 'is likely to', 'according to some experts', 'there is debate on'). "
        "Keep the information but with the correct level of certainty."
    ),
    severity="medium",
)
```

Because principles are data, extending the constitution means appending to a list — no subclassing, no code changes to the filter itself.

### Applying the filter

The notebook wraps the pattern in a small helper: build a `ConstitutionalFilter` with the chosen principles and a per-principle revision budget, then `apply()` it to the text along with optional context (the filter passes both to the LLM, always through `SecurePromptBuilder` — the evaluated text is untrusted input and is never f-stringed into the prompt):

```python
filt = ConstitutionalFilter(
    principles=principles,
    max_revisions=3,  # up to 3 revisions per principle
)

result = await filt.apply(
    output=text_sample["text"],
    context=text_sample["context"],
)
```

Internally, `apply()` iterates the principles in order. For each one it calls `check_principle()`; on a violation it asks the LLM to rewrite (`revision_prompt`), records a `ConstitutionalRevision` (original, revised, violation description), and re-checks the revised text. If a principle is still violated after `max_revisions` attempts, the loop moves on but the result flags `max_revisions_reached=True` and `all_principles_satisfied=False` — the pattern never silently claims success. Every applied revision also emits a `constitutional_revision_applied` structured log event, and the whole run is traced under an OTel `constitutional.apply` span.

### Reading the result

`ConstitutionalResult` is designed so downstream code can make a policy decision from the flags alone:

```python
print(f"All principles satisfied : {result.all_principles_satisfied}")
print(f"Max revisions reached    : {result.max_revisions_reached}")
for rev in result.revisions:
    print(f"[{rev.principle_id}] Violation: {rev.violation_detected[:80]}")
    print(f"  → Revised: {rev.revised[:120]}")
print(result.final_output)
```

A clean text passes with zero revisions; the PII-laden support reply typically triggers P003 and comes back anonymised; the false-history paragraph trips P002. The notebook's final section aggregates violations per principle with a `Counter` across all five texts — a miniature red-team report.

### Why the cap matters

`max_revisions_reached` is the honest failure mode: some texts *cannot* be revised into compliance (e.g. the entire content is the violation). Rather than looping or fabricating, the filter returns the best attempt and lets the caller decide — block, escalate to a human, or fall back to a refusal.

### Aggregate analysis across the dataset

The last part of the notebook runs the *combined* constitution (three defaults plus the two custom principles) over all five texts and tallies which rules fire, turning the filter into a lightweight red-team harness:

```python
all_principles = (DEFAULT_PRINCIPLES or []) + CUSTOM_PRINCIPLES

violations_by_principle: Counter = Counter()
for _, r in all_results:
    for rev in r.revisions:
        violations_by_principle[rev.principle_id] += 1

fully_safe = sum(1 for _, r in all_results if r.all_principles_satisfied)
```

This per-principle breakdown is exactly what you would track over time in production: a spike in P003 (PII) violations points at an upstream data-handling bug, while a rise in P002 suggests the base model or a prompt change is producing more confident falsehoods.

## Key API

| Symbol | Role |
|---|---|
| `ConstitutionalFilter` | Applies a sequence of principles to a text; `apply(output, context=None)` returns a `ConstitutionalResult`. Configured with `principles`, `max_revisions` (≥ 1), optional `settings`. |
| `ConstitutionalPrinciple` | Dataclass defining one rule: `id`, `name`, `description`, `critique_prompt`, `revision_prompt`, `severity` (`critical`/`high`/`medium`). |
| `ConstitutionalResult` | Outcome: `final_output`, `revisions`, `principles_checked`, `all_principles_satisfied`, `max_revisions_reached`. |
| `ConstitutionalRevision` | One audit record: `principle_id`, `original`, `revised`, `violation_detected`. |
| `DEFAULT_PRINCIPLES` | Built-in constitution: P001 harm, P002 factual accuracy, P003 PII exposure. |
| `SecurePromptBuilder` | Used internally so the evaluated text (untrusted content) never gets concatenated into prompt templates. |
| `ProviderRegistry` / `get_settings` | Resolve the LLM backend the filter uses for critique and revision calls. |

## Run it

```bash
uv run jupyter lab notebooks/patterns/07_constitutional_ai.ipynb
# or the script version:
uv run python examples/patterns/07_constitutional_ai.py   # from the prismal repo
```

Requires an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env` — critique and revision are real model calls.

## Related

- [Reflection Loop](06_reflection_loop.md) — the general generate → critique → refine primitive; Constitutional AI is its policy-driven cousin with explicit written rules.
- [Debate](02_debate.md) — another way to stress-test an answer, via adversarial agents instead of fixed principles.
- [Kokoro deliberation](../kokoro_deliberation.md) — persona-driven deliberation with a judge rendering the accountable decision.
- [Budget governance](../budget_governance.md) — the revise/check loop makes multiple LLM calls; budget guards keep that spend bounded.
