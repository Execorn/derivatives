# Contributing to DeepVol / Derivatives

Thank you for your interest in contributing to `Execorn/derivatives`! This quantitative library requires rigorous mathematical correctness and zero arbitrage tolerance.

## Development Workflow

1. **Environment Setup**:
   ```bash
   uv sync --all-extras
   ```
2. **Coding Standards**:
   - Python 3.9+ with strict type annotations (`mypy` / `jaxtyping`).
   - Clean architecture, no machine-specific hardcoded paths.
   - All Greeks must be analytically verified against closed-form benchmarks or central-bump oracles.
3. **Running Tests**:
   ```bash
   uv run pytest tests/test_eq_autocall_*.py
   uv run pytest tests/audit -m audit
   ```
4. **Lean 4 Proofs**:
   ```bash
   cd formal_proofs
   lake build
   ```
5. **Pull Requests**:
   - Ensure all tests pass.
   - Provide mathematical rationale for any pricing or Greek amendments.
