---
name: mining-quant-literature
description: >-
  Automated ingestion, extraction, and synthesis of quantitative finance literature from arXiv,
  SSRN, OpenAlex, and Crossref. Extracts SDEs, PDEs, payoffs, and calibration schemes into
  structured LaTeX tuples and constructs bipartite citation-assumption knowledge graphs for the deepvol framework.
---

# Quantitative Literature Mining & Research Extraction (`quant-literature-miner`)

This skill defines the autonomous protocol for ingesting, parsing, and synthesizing quantitative mathematical literature across `arXiv` (categories `q-fin.PR`, `q-fin.CP`, `q-fin.MF`, `math.PR`, `stat.ML`, `cs.LG`), `OpenAlex`, `Crossref`, and local research libraries.

---

## 1. Literature Discovery Pipeline & Search Directives

When exploring SOTA developments for derivatives pricing, rough volatility, or neural operators, the agent utilizes `arxiv-mcp-server` and `openalex` with targeted boolean queries:

### Canonical Search Queries
- **Rough Volatility & Hybrid Schemes**:
  `ti:"rough Bergomi" OR abs:"hybrid scheme" AND abs:"Volterra"`
- **Neural Operators in Finance**:
  `ti:"neural operator" OR ti:"Fourier neural" AND abs:"implied volatility" OR abs:"option pricing"`
- **Signature SDEs & Path-Dependent PDEs**:
  `ti:"signature SDE" OR abs:"path signature" AND abs:"path-dependent PDE" OR abs:"exotic"`
- **McKean-Vlasov & Particle Calibration**:
  `ti:"McKean-Vlasov" OR abs:"particle method" AND abs:"local stochastic volatility"`
- **High-Frequency & Dilated Convolutions**:
  `ti:"DeepVol" OR abs:"dilated causal convolutions" AND abs:"volatility forecasting"`

---

## 2. PDF-to-Markdown Extraction & LaTeX Block Preservation

1. **Retrieval**: Download the paper via `download_paper` from `arxiv-mcp-server` into `${WORKSPACE_FOLDER}/data/arxiv_library/`.
2. **Parsing**: Convert the manuscript into structured Markdown, strictly preserving all LaTeX equation blocks (`$$...$$` and `$...$`).
3. **Canonical Tuple Extraction**:
   Extract all stochastic dynamics into standard 5-tuples:
   $$(\mathcal{X}, \;\mu(t, X_t), \;\Sigma(t, X_t), \;\mathcal{F}_t, \;\mathbb{Q})$$
   - $\mathcal{X} \subseteq \mathbb{R}^d$: State space manifold and boundary conditions.
   - $\mu(t, X_t)$: Instantaneous drift under pricing measure $\mathbb{Q}$.
   - $\Sigma(t, X_t)$: Volatility dispersion / diffusion tensor.
   - $\mathcal{F}_t$: Underlying filtration (Brownian, fractional Brownian, jump-diffusion).
   - $\mathbb{Q}$: Risk-neutral pricing measure and numeraire definition.

---

## 3. Bipartite Citation-and-Assumption Knowledge Graph

Every extracted model must be cataloged in a bipartite graph linking:
* **Mathematical Assumptions**: Martingality, Novikov condition $\mathbb{E}[e^{\frac{1}{2}\int_0^T |\theta_t|^2 dt}] < \infty$, Durrleman absence of butterfly arbitrage ($g(k) \ge 0$), Feller condition ($2\kappa\theta > \xi^2$).
* **Derivative Payoff Classes**: European vanillas, Worst-Of Autocallables, Bermudan cancellable swaptions, path-dependent barriers.

```
[ArXiv Preprint / Paper]
        │
        ├──> Extracted SDE Tuple (X, mu, Sigma, F_t, Q)
        │
        ├──> Core Assumptions (Girsanov integrability, Feller condition, H in (0, 1/2))
        │
        └──> Mathematical Target (Backward Kolmogorov PDE / Infinitesimal Generator)
                │
                ▼
     (Passed to `verifying-symbolic-sdes`)
```

---

## 4. Subagent Operational Workflow

When spawned as a `quant-literature-miner` subagent in Google Antigravity:
1. Query `arxiv` / `paper-search` for papers published from 2022 to present matching the research topic.
2. Ingest the top 5 papers, parse their methodological innovations, and extract exact parameter definitions.
3. Output a structured literature review artifact in `<appDataDir>/brain/<conversation-id>/literature_synthesis.md`.
4. Route derived SDEs and PDE generator terms directly to `verifying-symbolic-sdes` for formal verification in Wolfram Engine 15.0 and Lean 4.
