# Agent Identity: Verifiable DIDs, Scoped Secrets, and Least-Privilege Delegation

> **Notebook:** [`notebooks/agent_identity.ipynb`](../notebooks/agent_identity.ipynb) · **Example:** [`examples/agent_identity.py`](https://github.com/prismal-ai/prismal/blob/main/examples/agent_identity.py)

Most agent frameworks run every agent as the same ambient process: one shared set of environment secrets, one implicit "allowed to do anything" identity. Phase IDN gives each prismal agent a **verifiable identity of its own** — a W3C `did:key` backed by a real Ed25519 keypair — plus least-privilege `Scope`s, a vault that releases secrets only to holders of the right scope, a default-deny `PolicyEngine` for authorizing actions, and on-behalf-of delegation tokens whose scopes can only *narrow* as they propagate down a chain of sub-agents. It is the "who is acting, and with whose authority?" layer that complements Phase H's "is this action safe?" — in fact the identity engine delegates its `tool_call` decisions to the Phase H `ToolPolicyEngine` so the two policy planes can never disagree.

The layer is opt-in (`identity_enabled`, default `False`) and, like every prismal port, ships deterministic fakes. This notebook drives all four mechanisms with `LocalIdentityProvider` (real cryptography, zero network — `did:key` resolution is fully offline) and a `FakeVault`, so no IdP, LLM, or API key is involved.

## What it demonstrates

- Minting a verifiable per-agent identity — a genuine Ed25519 `did:key` — with least-privilege scopes, then verifying it.
- Storing a secret in a vault and resolving it under the correct scope, with the raw value locked inside a `SecretStr`; an out-of-scope resolve raises `ScopeError`.
- Authorizing tool calls through the identity `PolicyEngine`: in-scope allowed, destructive routed to HITL, everything unmatched denied by default.
- Minting an on-behalf-of token for a user and propagating it to a sub-agent with **narrowed** scopes — widening is structurally impossible.

## How it works

### Minting a verifiable agent identity

`LocalIdentityProvider.issue()` generates a fresh Ed25519 keypair and derives the DID from the raw public key (multicodec + base58btc, per the `did:key` method) — self-contained and resolvable offline. The returned `AgentIdentity` is frozen and carries the DID, the agent name, and its granted scopes; it never carries key material or secrets:

```python
provider = LocalIdentityProvider()
coder = provider.issue(agent_name="coder", scopes=(Scope("tools:write_file"),))
print(coder.did)                    # did:key:z6Mk...
print(provider.verify(coder.did))   # True — known and not revoked
```

`verify()` answers "did I issue this, and is it still valid?" (revocation is a `provider.revoke(did)` away), and `provider.sign(did, payload)` produces real signatures for cross-boundary proof. For A2A federation the same settings can derive a `did:web` instead, which lands in the published Agent Card; tests use `FakeIdentityProvider`, which mints stable `did:fake:...` strings with no crypto at all.

### Scoped secrets that never leave the vault in the clear

A `Scope` is a glob over a resource string (`tools:write_file`, `net:egress`, `rag:*`), matched with `fnmatchcase`. Vaults bind secrets to scopes at `store()` time and enforce coverage at `resolve()` time — a caller must present scopes covered by the grant, or the resolve fails with a typed `ScopeError`:

```python
vault: FakeVault | EnvVault = FakeVault()
vault.store("openai", SecretStr("sk-secret"), scopes=(Scope("net:egress"),))

cred = vault.resolve("openai", scopes=(Scope("net:egress"),))
print(cred.ref, cred.value)     # 'openai' SecretStr('**********')  — masked

vault.resolve("openai", scopes=(Scope("tools:write_file"),))
# -> ScopeError: file-writing authority does not buy network credentials
```

The resolved `Credential` exposes an opaque `ref` and a `SecretStr` value — printing it yields asterisks, so a secret cannot leak through a log line or an agent transcript by accident. Identities themselves store only `credential_ref`, never values. `EnvVault` is the production sibling that reads through the injected `ConfigSourcePort` (Phase W — never `os.environ` directly); `FileVault` adds Fernet encryption at rest.

### Default-deny authorization with HITL escalation

`PolicyEngine.allow()` evaluates frozen `IdentityPolicy` rules (glob patterns over identity, action, and resource). When several match, the most specific wins; when **nothing** matches, the decision is `DENY` — least privilege is the floor, not an opt-in:

```python
engine = PolicyEngine([
    IdentityPolicy("coder", "tool_call", "tools:write_file",
                   PolicyEffect.ALLOW, "tools:write_file"),   # and must hold the scope
    IdentityPolicy("coder", "tool_call", "tools:delete_file", PolicyEffect.REQUIRE_HITL),
    IdentityPolicy("*", "tool_call", "*", PolicyEffect.DENY),  # explicit catch-all
])
engine.allow(identity=coder, action="tool_call", resource="tools:write_file")
# -> allow      · matched rule, and coder's scopes cover the require_scope
engine.allow(identity=coder, action="tool_call", resource="tools:delete_file")
# -> require_hitl · destructive op pauses for a human
engine.allow(identity=coder, action="tool_call", resource="secrets:db")
# -> deny       · caught by the catch-all (and by default-deny regardless)
```

Note the double lock on the `ALLOW` rule: matching the policy is not enough — its `require_scope` must also be covered by the identity's own scopes, binding *policy* to *credential*. `REQUIRE_HITL` routes through the existing `hitl_gate()` interrupt, and for `action="tool_call"` the engine first consults the Phase H `ToolPolicyEngine`, honoring its denials and HITL escalations. In production, rules load from `config/identity_policies.yaml` and evaluation happens pre-action at the `ActionInterceptor` seam.

### On-behalf-of delegation that can only narrow

When an agent acts *for a user* and fans work out to sub-agents, authority must attenuate, never amplify. `mint_on_behalf()` creates a TTL-bounded token rooted at the issuing agent's DID; `propagate()` extends the chain one hop and may only *shrink* the scope set — requesting a scope the token does not already cover raises `DelegationError`:

```python
token = mint_on_behalf(
    subject="user@example.com", scopes=(Scope("rag:*"),), ttl_s=900, issuer=coder.did
)
narrowed = propagate(token, via="did:key:zSubAgent", scopes=(Scope("rag:read"),))
print(narrowed.chain)   # (coder's did, 'did:key:zSubAgent') — full provenance

validate(narrowed, action="tool_call", resource="rag:read")    # True
validate(narrowed, action="tool_call", resource="rag:write")   # False — narrowed away
```

`validate()` never raises: it returns `False` for revoked tokens, expired TTLs, or uncovered resources. The `chain` records every DID the delegation passed through, giving the audit log a complete answer to "who authorized this, on whose behalf, via whom?". Only the serializable resource strings thread through `state["metadata"]["identity"]["on_behalf"]`; revocation lives in an in-process set, never in checkpointed state.

Taken together the four mechanisms compose into one governance story: an agent *is* someone (`did:key`), *holds* only the authority it was granted (scopes), *proves* that authority to get secrets (vault) and permission to act (policy engine), and *hands down* strictly less authority to anything acting beneath it (delegation). Each piece degrades safely on its own — default-deny policies, `ScopeError` on over-reach, `False` on stale tokens — so a misconfiguration fails toward less privilege, not more.

## Key API

| Symbol | Role |
|---|---|
| `LocalIdentityProvider` | Issues/verifies/revokes real Ed25519 `did:key` identities in-process; `sign()` for cross-boundary proof |
| `AgentIdentity` / `Scope` | Frozen identity (DID, agent name, scopes, `credential_ref`) and glob-matched resource scope |
| `FakeVault` / `EnvVault` | Scope-enforcing secret stores; resolve returns a `Credential` with a masked `SecretStr`, out-of-scope → `ScopeError` |
| `IdentityPolicy` / `PolicyEngine` / `PolicyEffect` | Glob rules, most-specific-wins, **default deny**; `ALLOW` additionally enforces `require_scope`; `REQUIRE_HITL` pauses for a human |
| `mint_on_behalf()` / `propagate()` / `validate()` | TTL-bounded delegation tokens; scopes narrow-only (widening raises `DelegationError`); `chain` records provenance |
| `ScopeError` | Typed rejection for scope-coverage violations (vault and delegation) |
| `Settings.identity_*` | `identity_enabled` (default `False`), `identity_mode` (`off`/`warn`/`enforce`), provider (`local`/`oidc`), DID method (`key`/`web`), vault backend, policy path, on-behalf TTL |

## Run it

```bash
uv run jupyter lab notebooks/agent_identity.ipynb
uv run python examples/agent_identity.py   # from the prismal repo
```

No API key is required — `did:key` issuance and resolution are fully offline, and the vault and policy engine are pure in-process objects. The notebook's final run cell ships commented out as `# main()`; since `main` is synchronous here, just uncomment it and run — no `await` needed.

## Related

- [Runtime hardening](runtime_hardening.md) — the Phase H `ToolPolicyEngine` this identity engine delegates tool-call decisions to
- [A2A server](a2a_server.md) — where the agent's DID surfaces in the published Agent Card for federation
- [Custom config sources](config_source_custom.md) — the `ConfigSourcePort` that `EnvVault` reads secrets through
- [Custom tool providers](tool_provider_custom.md) — the tool-injection seam whose calls these policies authorize
