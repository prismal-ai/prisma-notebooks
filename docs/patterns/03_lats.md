# LATS (Language Agent Tree Search)

> **Notebook:** [`notebooks/patterns/03_lats.ipynb`](../../notebooks/patterns/03_lats.ipynb) · **Example:** [`examples/patterns/03_lats.py`](https://github.com/prismal-ai/prismal/blob/main/examples/patterns/03_lats.py)

LATS (`prismal.agents.patterns.lats`, SPEC-PAT-004) applies Monte Carlo Tree Search to an agent's *action space*. Where Tree of Thoughts searches over reasoning text, LATS searches over what the agent should *do* next: each MCTS simulation selects a promising node via UCB1, expands it with candidate actions, scores the resulting state with a reward function, and backpropagates that reward up the tree. UCB1 (Auer et al. 2002, `Q/N + C·√(ln N_parent / N)`) is what balances exploiting high-reward branches against exploring rarely-visited ones.

The notebook runs `LATSAgent` on **WebArena-style planning tasks** (three embedded synthetic scenarios inspired by [WebArena](https://arxiv.org/abs/2307.13854)): find the cheapest flight, book a restaurant in under three steps, research and compare AI agent frameworks. Discrete tool actions, multi-step goals, and a reward that trades off information gathered against steps taken — exactly the setting where MCTS earns its keep. Notably, this demo runs *without an LLM in the loop*: the action generator and transition function are deterministic simulations, which makes the MCTS mechanics themselves easy to observe and reproduce.

## What it demonstrates

- Constructing a `LATSAgent` from three injected callables — `action_generator`, `transition_fn`, `reward_fn` — with the tree-search logic fully decoupled from any LLM or tool framework.
- Simulating a discrete action space (`search_web`, `extract_info`, `compare_options`, `summarize`, `confirm_action`, ...) and its state transitions.
- Designing a shaped reward in `[0, 1]`: credit for relevant information, a penalty for inefficiency, a completion bonus.
- Bounding the search with `max_simulations`, `max_depth`, `timeout_seconds`, and an early-stop `terminal_reward` threshold.
- Reading a `LATSResult`: the best action sequence, its average reward, and how much of the tree the search actually explored.

## How it works

### Three callables define the environment

`LATSAgent` never assumes how actions are produced or applied — you inject the environment. The action generator proposes candidate actions for the current state (in production this would be an LLM call; here it is a staged heuristic over `steps_taken`):

```python
async def action_generator(state: dict[str, Any], tools: list[dict]) -> list[str]:
    steps = state.get("steps_taken", 0)
    goal = state.get("goal", "")
    if steps == 0:
        return [
            f"search_web(query='{goal}')",
            f"search_web(query='{goal} price best option')",
            f"search_web(query='{goal} 2024 comparison')",
        ]
    if steps == 1 and not state.get("info_gathered"):
        return ["extract_info(source='first result')", "open_url(url='most relevant result')", ...]
    ...
```

The transition function applies an action and returns a *new* state (it must not mutate the input — MCTS revisits nodes, so states have to stay stable):

```python
async def transition_fn(state: dict[str, Any], action: str) -> dict[str, Any]:
    new_state = dict(state)
    new_info = list(state.get("info_gathered", []))
    if "search_web" in action:
        new_info.append(f"search_results:{action[:50]}")
    elif "summarize" in action and "flight" in state.get("goal", ""):
        new_info.append("summary:cheapest_flight_price_450EUR_airline_X")
    ...
    new_state["info_gathered"] = new_info
    new_state["steps_taken"] = state.get("steps_taken", 0) + 1
    return new_state
```

### Reward shaping steers the search

The reward function is the only signal MCTS optimizes, so its shape matters more than any other knob. This one rewards relevant information (+0.2 per item), penalizes wandering (−0.1 per step beyond 5), and pays a +0.5 bonus once a terminal summary appears:

```python
async def reward_fn(state: dict[str, Any]) -> float:
    score = 0.0
    for item in state.get("info_gathered", []):
        if any(kw in item.lower() for kw in {"price", "comparison", "confirmed", "summary", "options"}):
            score += 0.2
    if state.get("steps_taken", 0) > 5:
        score -= 0.1 * (state["steps_taken"] - 5)
    # +0.5 completion bonus when a terminal summary is present
    return max(0.0, min(1.0, score))
```

Because the bonus only fires for states containing a `summary` with a terminal marker, plans that summarize too early or too late score worse than plans that gather, compare, then conclude — the search discovers that ordering on its own.

### Configuring and running the search

```python
agent = LATSAgent(
    tools=AVAILABLE_TOOLS,
    reward_fn=reward_fn,
    action_generator=action_generator,
    transition_fn=transition_fn,
    max_simulations=30,          # MCTS iterations
    exploration_constant=1.41,   # UCB1's C = √2 (Auer et al. 2002)
    max_depth=5,                 # maximum tree depth
    timeout_seconds=30.0,        # wall-clock safety budget
    terminal_reward=0.95,        # nodes scoring >= this are marked terminal
)
result = await agent.search(initial_state=task["initial_state"], goal=task["goal"])
```

Each of the 30 simulations runs the four MCTS steps: **select** (descend by max UCB1 — unvisited nodes score `+inf`, so every fresh child gets tried once before any node is revisited), **expand** (a visited, non-terminal leaf within `max_depth` gets one child per candidate action), **simulate** (score the state with `reward_fn`; at or above `terminal_reward` the node is marked terminal), and **backpropagate** (increment visits and accumulate reward from the node back to the root). The search stops early on the wall-clock deadline, and raises `LATSError` only if no simulation ran at all or the generator never produced a child.

### Reading the result

Extraction is exploit-only: `search()` follows the child with the best *average* reward (`Q/N`) at each level, ignoring the exploration bonus that guided the search itself.

```python
print(f"Simulations: {result.total_simulations}")
print(f"Nodes explored: {result.nodes_explored}")
print(f"Best reward: {result.best_reward:.3f}")   # average reward of the best leaf
print(f"Task completed: {'Yes' if result.goal_reached else 'No'}")
for j, action in enumerate(result.best_action_sequence, 1):
    print(f"  {j}. {action}")
info = result.best_state.get("info_gathered", [])   # state reached by the best path
```

For the flight task the winning sequence typically reads `search_web → extract_info → compare_options → summarize` — gather, verify, compare, conclude — with the shaped reward near its ceiling. The closing cell of the notebook recaps why: the `C = √2` coefficient made the agent revisit strong branches (exploitation, `Q/N`) while still sampling under-visited ones (exploration, `√(ln N_parent / N)`), covering the action space far more efficiently than enumerating every 3-to-4-step plan.

## Key API

| Symbol | Role |
|---|---|
| `LATSAgent` | The MCTS engine; constructed from tools + the three callables, plus `max_simulations`, `exploration_constant`, `max_depth`, `timeout_seconds`, `terminal_reward`. |
| `LATSAgent.search()` | Async entry point `(initial_state, goal) -> LATSResult`; raises `LATSError` on a vacuous search. |
| `LATSResult` | Dataclass: `best_action_sequence`, `best_state`, `total_simulations`, `nodes_explored`, `best_reward`, `goal_reached`. |
| `LATSNode` | One MCTS tree node (`state`, `action`, `reward`, `visits`, `children`, `is_terminal`) with a `ucb1()` method. |
| `ActionGeneratorFn` | Callable contract `(state, tools) -> list of candidate actions`. |
| `TransitionFn` | Callable contract `(state, action) -> new_state` — must be pure. |
| `RewardFn` | Callable contract `(state) -> float in [0, 1]`. |

`search()` also accepts a `budget_guard_fn` so Budget Governance can soft-cap simulations mid-run (one simulation always runs first, keeping the search non-vacuous).

## Run it

```bash
uv run jupyter lab notebooks/patterns/03_lats.ipynb
# or the script version:
uv run python examples/patterns/03_lats.py   # from the prismal repo
```

The notebook header requires an API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in `.env`, per the shared setup — though this particular demo's callables are simulated, so the search itself makes no LLM calls.

## Related

- [Tree of Thoughts (ToT)](01_tree_of_thoughts.md) — the sibling tree search: beam/BFS/DFS over reasoning steps instead of MCTS over actions.
- [Debate / Society of Mind](02_debate.md) — a different route to better answers: multi-agent argument rather than search.
- [Budget Governance](../budget_governance.md) — how `budget_guard_fn` caps expensive patterns like LATS at runtime.
