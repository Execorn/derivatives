# GLOBAL CONTEXT: Deep Learning Calibration of Heston Model

## Mission
We are rewriting and modernizing the code from Horvath et al. (2019) "Deep Learning Volatility".
The original code uses outdated TensorFlow/Keras.
Our goal is to build a modern, clean, object-oriented PyTorch pipeline and save it in the `src/` directory.

## Repository Structure
- Original Data: `data/HestonTrainSet.txt.gz`
- Target Output Directory: `src/`

## Mathematical Constraints
- Inputs (5 Heston params): kappa, theta, sigma, rho, v0.
- Outputs (Implied Volatility Surface): Flattened grid of size 88 (8 maturities x 11 strikes).
- Feller Condition constraint: 2 * kappa * theta > sigma^2.

## Technology Stack
- Arch Linux environment.
- Python 3.10+, PyTorch 2.0+, scikit-learn, scipy.
- **Strict Rule for NN:** DO NOT use ReLU. Use `nn.ELU()` to ensure C^2 smoothness.
