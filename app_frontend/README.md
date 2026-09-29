# DeepVol Institutional Frontend

Next.js 16+ web application for interactive volatility surface exploration, real-time derivative risk telemetry, and structured autocallable note pricing.

---

## Capabilities

### 1. Implied Volatility Surface & Risk Telemetry (`/`)
* **3D WebGL Surface Visualization**: Three.js interactive canvas rendering parameterized implied volatility surfaces across strike and maturity dimensions.
* **Real-Time Calibration**: Interface to trigger Gauss-Newton and Levenberg-Marquardt calibration against SPX or Deribit cryptocurrency market data.
* **WebSocket Risk Stream**: Connects to the FastAPI backend (`/ws/risk`) with conflation queuing to visualize live streaming Greeks (Delta, Gamma, Vega) without interface lag.
* **Scenario Stress Testing**: Evaluates spot price and volatility shocks with real-time portfolio revaluation.

### 2. Autocallable Structured Notes (`/autocall`)
* **Multi-Fidelity Pricing**: Combines a Gatheral Crank-Nicolson PDE base solver with a 5-member residual MLP ensemble.
* **Epistemic Uncertainty Gauge**: Displays model uncertainty $\sigma_{\text{bps}}$ with calibrated threshold gating ($\tau_{\text{OOD}} = 2.17$ bps).
* **Barrier Sensitivity Analysis**: Quantifies discontinuous barrier risks including Autocall Barrier Delta ($\partial V / \partial B_{\text{autocall}}$) and Dip Barrier Delta ($\partial V / \partial B_{\text{dip}}$).
* **SR 26-2 Compliance Telemetry**: Real-time auditing of input parameter validity, population drift, and automatic fallback routing to GPU Monte Carlo or analytical PDE.

---

## Setup & Execution

### Prerequisites
* Node.js 18.17+ or Node.js 20+
* Running DeepVol FastAPI backend on port 8000:
  ```bash
  uvicorn deepvol.api.server:app --reload --port 8000 --app-dir src
  ```

### Development Server
```bash
npm install
npm run dev
```

The application will be accessible at:
* Volatility Surface & Risk: [http://localhost:3000](http://localhost:3000)
* Structured Autocallable Notes: [http://localhost:3000/autocall](http://localhost:3000/autocall)

### Production Build
```bash
npm run build
npm start
```
