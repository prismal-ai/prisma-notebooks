# Skill Creator Agent: generate, validate, and gate new skills offline

> **Notebook:** [`notebooks/skill_creator_agent.ipynb`](../notebooks/skill_creator_agent.ipynb) · **Example:** [`examples/skill_creator_agent.py`](https://github.com/prismal-ai/prismal/blob/main/examples/skill_creator_agent.py)

`SkillCreatorAgent` (`prismal/agents/skill_creator.py`) turns a natural-language specification
into a complete Prismal skill: an LLM writes `skill.py`, a shell-execution guard screens the
output, ruff/mypy/bandit validate it (with an LLM fix loop on failure), and the artefacts land in
`skills/custom/<slug>/` behind a human-review sentinel. It is one of the 26 specialist agents
behind the supervisor — the `skill_creator` route — and it embodies prismal's skill-tier policy:
AI-generated code goes to the gitignored `custom/` tier and cannot be activated until a human
renames the sentinel to `validated_by_human.txt`.

The notebook runs the whole five-step workflow offline by injecting a deterministic fake at the
module's single LLM seam, `_llm_generate`. Everything else — the guard, the quality tools, the
artefact writing, the LangGraph node — is the real code path, and all output goes to a temporary
directory, never into `prismal/skills/`.

## What it demonstrates

- The five-step workflow: GENERATE → shell guard (AC-017-8) → VALIDATE (ruff + mypy + bandit with
  a bounded fix loop) → WRITE artefacts → REPORT.
- Injecting a fake at the `_llm_generate` seam so the pipeline runs without a provider — the same
  factory-injection style used across prismal's testable modules.
- The AC-017-8 refusal: a spec whose generated code contains shell/subprocess patterns is rejected
  outright unless the caller passes `allow_shell=True`.
- The four artefacts written per skill: `skill.py`, `human_review_required.txt`,
  `generation_prompt.txt` (the original spec, kept for audit), and `quality_report.json`.
- `skill_creator_node`, the LangGraph entry point the supervisor routes to, extracting the spec
  from the last human message.

## How it works

### The fake code-generation LLM

`_llm_generate(messages)` is the module's only LLM call; by default it lazily wires
`ProviderRegistry().get_llm()`. The demo replaces it with a coroutine that returns canned source —
a valid unit-converter skill for the normal spec, and deliberately subprocess-laden code for the
shell spec:

```python
async def fake_llm_generate(messages: list[dict[str, str]]) -> str:
    """Deterministic stand-in for the code-generation / fix-loop LLM calls.

    Picks the canned source by inspecting the (SecurePromptBuilder-built) user
    message; a real run would return freshly generated or fixed code instead.
    """
    user_content = messages[-1]["content"]
    if "shell" in user_content.lower():
        return FAKE_SHELL_CODE
    return FAKE_SKILL_CODE

skill_creator_module._llm_generate = fake_llm_generate
skill_creator_module.MAX_FIX_ITERATIONS = 1  # fake always returns the same code
```

Note the security seam the docstring points at: `create_skill()` never f-strings the user's spec
into a raw prompt. It builds the generation and fix messages through `SecurePromptBuilder`
(`builder.build(system=..., user=...)`), which isolates user-controlled content from the system
instructions — the framework-wide rule for anything user-authored. Capping
`MAX_FIX_ITERATIONS` at 1 just keeps the demo fast, since re-generating identical code would only
re-run the quality tools.

### Generation, validation, and the fix loop

The first demo generates a real skill into the temp root:

```python
with tempfile.TemporaryDirectory(prefix="skills_") as tmp:
    skills_root = Path(tmp)

    result = await create_skill(SPEC, skills_root=skills_root)
    print(result)

    skill_dirs = sorted(p for p in (skills_root / "custom").iterdir() if p.is_dir())
    for skill_dir in skill_dirs:
        for artefact in sorted(skill_dir.iterdir()):
            print(f"  - custom/{skill_dir.name}/{artefact.name}")
```

After stripping any accidental Markdown fences, the agent derives the directory slug from the
generated `BaseSkill` subclass name, writes `skill.py`, and runs the three quality tools as real
subprocesses: `ruff check`, `mypy --ignore-missing-imports --disallow-untyped-defs`, and
`bandit -r -ll`. If any tool fails, the errors and the current code are fed back to the LLM as a
fix prompt (again via `SecurePromptBuilder`) and the loop retries, up to `MAX_FIX_ITERATIONS`
times (default 3). The tools run for real when installed locally; a missing tool simply shows up
as a failed check in the report rather than crashing the run. The final REPORT string summarizes
pass/fail per tool, iterations used, and the next steps — ending with the sentinel rename and
`prismal skills enable <slug>`.

### The shell-execution guard (AC-017-8)

Before anything is written to disk, the generated source is scanned for shell/subprocess patterns
(`subprocess.run`, `subprocess.Popen`, `popen`, and friends). Unless the caller opted in with
`allow_shell=True`, the agent refuses and returns a warning instead of a skill:

```python
refusal = await create_skill(SHELL_SPEC, skills_root=skills_root)
print(refusal)
# Refused: the generated skill contains shell/subprocess calls. ...
```

This is a policy gate on the *output*, complementing the human gate on activation: even a spec
that sounds harmless is rejected if the code the model produced would shell out. Note the guard
inspects the generated code, not the spec text — which is why the demo's `FAKE_SHELL_CODE`
contains a real `subprocess.run` call for it to catch.

### The human-review gate and audit artefacts

Every generated skill ships with three companions to `skill.py`. The sentinel
`human_review_required.txt` blocks activation until a reviewer renames it to
`validated_by_human.txt`; `generation_prompt.txt` preserves the original spec with a timestamp so
there is an audit trail from artefact back to request (AC-017-7); and `quality_report.json`
records each tool's pass/fail and output plus the number of fix iterations used (AC-017-4). All
of it lives under the `custom/` tier — gitignored by design, separate from the committed
`available/` tier and the runtime-enabled `active/` tier.

### The LangGraph node

Finally the notebook calls the same entry point the supervisor uses. `skill_creator_node` takes
an `AgentState`, pulls the spec from the most recent human message, delegates to `create_skill()`,
and returns the standard node update:

```python
skill_creator_module._SKILLS_CUSTOM_DIR = skills_root / "custom"
state = {"messages": [HumanMessage(content=SPEC)]}
update = await skill_creator_node(state)
print(update["current_agent"])          # 'skill_creator'
print(update["messages"][0].content)    # the REPORT string as an AIMessage
```

Because the node has no `skills_root` parameter (in the graph it always writes to the module
default), the demo points `_SKILLS_CUSTOM_DIR` at the temp directory too. The returned dict —
new `messages` plus `current_agent` — is the contract every specialist node honors before control
returns to the supervisor.

## Key API

| Symbol | Role |
|---|---|
| `create_skill(spec, skills_root=None, allow_shell=False)` | The five-step workflow; returns a multi-line report string (or a refusal for shell-pattern code). |
| `skill_creator_node(state)` | LangGraph node behind the supervisor; extracts the spec from the last human message and delegates to `create_skill`. |
| `MAX_FIX_ITERATIONS` | Cap on LLM fix rounds after failed validation (default 3; set to 1 in the demo). |
| `_llm_generate(messages)` | The module's single LLM seam; defaults to `ProviderRegistry().get_llm()`, replaced by the fake here. |
| `SecurePromptBuilder` | Builds the generation and fix prompts so the user spec never gets f-stringed into system instructions. |

## Run it

```bash
uv run jupyter lab notebooks/skill_creator_agent.ipynb
uv run python examples/skill_creator_agent.py   # from the prismal repo
```

No API key required — the demo runs fully offline with an injected fake LLM (ruff/mypy/bandit run for real if installed).

## Related

- [MetaLearner](meta_learner.md) — proposes the new skills this agent generates, behind the same human-review discipline.
- [Supervisor quickstart](supervisor_quickstart.md) — how the supervisor routes to specialist nodes like `skill_creator`.
- [Custom tool provider](tool_provider_custom.md) — how activated skills actually reach agents, via the injected `ToolProviderPort`.
- [Blind review pipeline](blind_review_pipeline.md) — another pipeline built around gated, multi-stage code review.
