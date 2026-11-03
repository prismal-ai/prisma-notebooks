# LLM Gateways III: OmniRoute and the Fail-Closed Startup Gate

> **Notebook:** [`notebooks/gateways_omniroute.ipynb`](../notebooks/gateways_omniroute.ipynb) · **Example:** [`examples/gateways_omniroute.py`](https://github.com/prismal-ai/prismal/blob/main/examples/gateways_omniroute.py) · **Guide:** [`docs/gateways.md`](https://github.com/prismal-ai/prismal/blob/main/docs/gateways.md)

OmniRoute is a **self-hosted** gateway, so prismal does not trust it until a startup gate passes. The gate has three requirements:

- an API key;
- an explicit upstream attestation (`omniroute_upstream_attestation`, e.g. `api_keys_only`);
- a version floor that excludes releases affected by CVE-2026-88062.

The gate runs in `GatewayRouter.astartup()` against `/api/monitoring/health`. If the gate fails, or crashes, the gateway is disabled and every `resolve()` for it is refused.

## What it demonstrates

1. **Version 3.8.50 (vulnerable).** `router.disabled == {"omniroute": "omniroute_vulnerable_version"}`, and `resolve("gw:omniroute/my-combo")` raises `GatewayPolicyError`.
2. **Version 3.8.51 (fixed).** The gate passes and the combo resolves to `openai/my-combo` on the local `api_base`.

The health endpoint is an `httpx.MockTransport`, so the notebook makes no network calls.

Before `astartup()` has run, OmniRoute is also treated as disabled. Forgetting to start the router therefore can never fail open.
