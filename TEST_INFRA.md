# E2E Test Infrastructure & Methodology — Phase 7

## Overview
This document outlines the testing philosophy, methodology, and infrastructure designed for the Phase 7 (P7) Multi-Asset Calibration Framework in the derivatives pricing repository. Our objective is to ensure that all pricing, calibration, and data loading modules are validated against rigorous standards of software quality and mathematical correctness.

## Test Philosophy
We adhere to a "real-world scenario-driven" testing philosophy. Option pricing and calibration models are highly sensitive to input parameters, boundary conditions, and floating-point accuracy. Therefore, our testing infrastructure must verify not only that the software runs without errors, but that it behaves in a mathematically sound manner, maintains put-call parity, handles extreme market conditions gracefully, and performs robustly in end-to-end financial workflows.

### Core Principles
1. **No Facade Implementations**: Stubs and engines must implement genuine mathematical logic (e.g., Black-Scholes, Bachelier normal pricing, Heston integration, and SABR Hagan equations) rather than returning static dummy values.
2. **Strict Input Validation**: Every entry point must validate inputs (e.g., positive spots, strikes, volatilities, and valid correlation coefficients) and raise appropriate exceptions (like `ValueError`) for invalid values.
3. **Four-Tier Testing Structure**:
   - **Tier 1: Feature Coverage**: Verify all core methods of all engines and loaders under typical parameters.
   - **Tier 2: Boundary & Corner Cases**: Stress-test edge cases such as zero/negative prices, empty arrays, out-of-bounds parameters, and extreme volatilities.
   - **Tier 3: Cross-Feature Combinations**: Validate physical invariants and parity relations across different features (e.g., Call-Put Parity, consistency across delta spaces).
   - **Tier 4: Real-World Application Scenarios**: Build realistic multi-step workflows representing actual quant trading or risk management operations.

---

## Test Architecture (Tiers 1–4)

### Tier 1: Feature Coverage
Ensures all 7 components operate correctly with standard parameters. We run at least 5 distinct test cases per component (>=20 tests total):
- **Equity MLSV Engine**: Option pricing, conditional expectation E[V_t | S_t], and local vol grid calibration.
- **FX SABR Engine**: Delta-to-strike conversion, SABR beta=1 parameter calibration, and strike-vol grid extraction.
- **Rates LMM-SABR Engine**: Bachelier normal option pricing, displaced SABR option pricing, and vol cube trilinear interpolation.
- **Commodity Schwartz-Smith Engine**: Schwartz-Smith pricing, Heston characteristic function solver pricing, and comparison errors.

### Tier 2: Boundary & Corner Cases
Checks robustness against bad inputs, empty data, or edge conditions (>=20 tests total):
- Spots, strikes, volatilities, maturities, and parameters <= 0.
- Out-of-bounds correlation coefficients (rho outside [-1, 1]).
- Out-of-bounds delta for FX (delta outside valid domain of Garman-Kohlhagen).
- Empty or mismatching grids/surfaces for local vol calibration and swaption vol cube interpolation.
- Zero-volatility limits and extremely high-volatility limits.
- Invalid option types (other than "call" or "put").

### Tier 3: Cross-Feature Combinations
Validates consistency and mathematical parity rules (pairwise combinations):
- **Call-Put Parity**: Verified for all pricing engines (MLSV, Rates Bachelier, Displaced SABR, and Schwartz-Smith).
- **Strike Inversion Parity**: Verifies that converting delta to strike and pricing that strike yields a price consistent with the delta.
- **Model Comparison Convergence**: Checks that as short-term deviation decay in Schwartz-Smith goes to infinity or long-term variance dominates, option prices align in predictable bounds with Heston pricing.

### Tier 4: Real-World Application Scenarios
Simulates production workflows (>=5 scenario tests):
1. **Equity MLSV SPX Calibration**: Full workflow loading SPX data, building a Dupire local vol grid, and calibrating the McKean-Vlasov SDE solver.
2. **FX EUR/USD SABR Delta-Strike smile calibration**: Loading market FX quotes, converting deltas to strikes, calibrating SABR parameters, and extracting a high-resolution strike-volatility grid.
3. **SOFR Swaption Cube Loading and Pricing**: Loading the swaption cube, extracting slices, interpolating volatilities for custom tenors/expiries/strikes, and pricing swaptions via the Bachelier engine.
4. **CME WTI Crude Oil Schwartz-Smith Calibration**: Loading WTI futures curves, pricing options, and comparing pricing errors against Heston parameters to quantify model discrepancy.
5. **Multi-Asset Parallel Pricing**: Executing a simulated portfolio pricing run containing FX, Rates, Commodity, and Equity options in parallel.

---

## Test Environment & Execution

### Environment Setup
All tests are executed within the virtual environment located at `.venv/`. Dependencies include:
- `numpy`
- `scipy`
- `pytest`

### Executing the Test Suite
A dedicated test runner script is provided at `scripts/run_e2e_tests.sh`.

```bash
# To run the test suite
bash scripts/run_e2e_tests.sh
```

The script performs the following:
1. Activates the virtual environment.
2. Sets `PYTHONPATH` to include the project root.
3. Runs `pytest` on `tests/test_e2e_phase7.py` with verbose output.
