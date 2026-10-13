# Security & Supply Chain Audit Report (Phase 16)

**Audited Commit**: `f7f0dc4`
**Auditor**: API-Security-Auditor
**Date**: 2026-10-13T21:46:00Z

---

## 1. Checkpoint Deserialization & Supply Chain (Phase 16.1 & 16.2)

### 1.1 Unsafe PyTorch Deserialization (Finding F001 - Severity S0)
- **Vulnerability**: Python `pickle` deserialization via `torch.load` without `weights_only=True`.
- **Affected Locations**:
  1. `src/deepvol/api/server.py:2208`:
     ```python
     new_state = torch.load(weights_path, map_location="cpu")
     ```
  2. `src/deepvol/mrm/fx_cross_guardian.py:162`:
     ```python
     ckpt = torch.load(p, map_location="cpu")
     ```
  3. `src/deepvol/training/train_cross_surrogate.py:507`:
     ```python
     ckpt = torch.load(path, map_location="cpu")
     ```
  4. `src/deepvol/training/train_cross_surrogate.py:1032, 1040`:
     ```python
     ref_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
     m_ckpt = torch.load(ckpt_paths[m], map_location=eval_device)
     ```
- **Risk Assessment**: If an attacker or malicious insider tampers with a checkpoint or provides a crafted payload during reload, Python pickle execution enables arbitrary code execution (RCE) with the server process permissions.
- **Specification Mandate**: SPEC-DEEPVOL-FX-CROSS-003 §5 Phase 16.2 explicitly requires:
  `Checkpoints load with weights_only=True. Model paths cannot be traversed or injected from requests or environment variables.`
- **Remediation**: Pass `weights_only=True` to all `torch.load` invocations. Note: PyTorch 2.6+ defaults to `weights_only=True`, but explicit declaration guarantees defense across environments.

### 1.2 Path Traversal & Injection Analysis
- In `src/deepvol/api/server.py`:
  - `reload_fx_cross_weights(weights_path)` accepts `weights_path`. In the HTTP router, no public endpoint directly exposes `reload_fx_cross_weights` with user-supplied path query strings. It is intended for internal administration/signal handlers.
  - Checkpoint discovery in `_init_fx_cross_state()` uses resolved root paths (`get_project_root() / "artifacts/checkpoints/cross_surrogate_ensemble.pt"`).
  - Path traversal risk from HTTP requests is currently **LOW** because endpoints `/fx_cross/price`, `/fx_cross/price_batch`, `/fx_cross/health` accept only structured financial contract payloads, not filesystem paths.

### 1.3 Dependency Vulnerability Scan (pip-audit)
- `pip-audit` scanned the complete virtual environment and identified **111 vulnerabilities across 23 installed packages**.
- **Key Findings**:
  - `torch 2.8.0`: Multiple CVEs (PYSEC-2025-206, GHSA-vgrw-7cvw-pwgx). Upstream fix in torch 2.9.0 / 2.10.0.
  - `starlette 0.49.3` / `aiohttp 3.13.5`: HTTP parsing and DoS advisories.
  - `pyarrow 21.0.0`: PYSEC-2026-113.
  - `orjson 3.11.5`: GHSA-hx9q-6w63-j58v.
- **Triage**: In an isolated on-premises trading environment without untrusted public web traffic, the risk of high-level library CVEs is mitigated by perimeter controls. However, for containerized production serving, base images must bump patched minor versions where ABI-compatible.

---

## 2. Secrets & Credential Leakage Audit (Phase 16.3)
- Grepped entire codebase for high-entropy strings, API keys, private keys, AWS/cloud tokens, passwords, and private certificates:
  - 0 hardcoded private credentials or API keys found in `src/deepvol/`.
  - Logging statements in `server.py`, `fx_cross_runtime.py`, and `fx_cross_guardian.py` log only numerical metadata, shapes, latencies, and request identifiers. No PII, customer identifiers, or secrets are emitted.

---

## 3. Network Binding & CORS Configuration (Phase 16.5)
- In `src/deepvol/api/server.py`:
  - CORS middleware is configured:
    ```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Development default
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```
  - **Risk**: `allow_origins=["*"]` with `allow_credentials=True` is an insecure CORS configuration for production environments. For production deployment, allowed origins must be restricted to authenticated desk domains.
  - Server CLI default binds to `0.0.0.0` (all interfaces, Bandit B104). Production recommendation: bind to `127.0.0.1` or internal VPC interface behind an mTLS reverse proxy.
