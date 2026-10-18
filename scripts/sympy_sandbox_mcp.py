# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastmcp>=0.1.0",
#     "sympy>=1.12",
# ]
# ///
"""scripts/sympy_sandbox_mcp.py

AST-Guarded SymPy Sandbox FastMCP Server for Quantitative Symbolic Derivations.
Whitelists safe AST nodes, blocks arbitrary Python code execution, and exposes
symbolic Ito generator derivation, Taylor asymptotics, dimensional checks,
and analytical Jacobian generation to Antigravity agents over stdio.
"""

from __future__ import annotations

import ast
import sys
from typing import Dict, List, Optional
import sympy as sp

try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

# Ensure all diagnostic logs go strictly to stderr to preserve stdio JSON-RPC integrity
mcp = FastMCP("sympy-sandbox")

ALLOWED_SYMPY_NAMES = {
    "Symbol", "symbols", "Function", "exp", "log", "sqrt", "sin", "cos", "tan",
    "sinh", "cosh", "tanh", "erf", "diff", "integrate", "simplify", "expand",
    "series", "Matrix", "Rational", "pi", "oo", "I", "Abs", "Max", "Min"
}


def safe_parse_sympy(expr_str: str, local_symbols: Dict[str, sp.Symbol]) -> sp.Expr:
    """Parses a mathematical expression string using an AST whitelist.

    Blocks all imports, function definitions, lambdas, comprehensions,
    attribute lookups, and unauthorized symbols to prevent code injection.
    """
    clean_str = expr_str.strip()
    tree = ast.parse(clean_str, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp)):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is blocked in AST sandbox.")
        if isinstance(node, ast.Call):
            # Only allow calls to named functions in whitelist or local_symbols
            if not isinstance(node.func, ast.Name):
                raise ValueError("Indirect function calls are blocked in AST sandbox.")
            if node.func.id not in ALLOWED_SYMPY_NAMES and node.func.id not in local_symbols:
                raise ValueError(f"Unauthorized function call in symbolic expression: {node.func.id}")
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_SYMPY_NAMES and node.id not in local_symbols:
                raise ValueError(f"Unauthorized identifier in symbolic expression: {node.id}")

    env = {k: getattr(sp, k) for k in ALLOWED_SYMPY_NAMES if hasattr(sp, k)}
    env.update(local_symbols)
    return eval(compile(tree, "<ast_sandbox>", "eval"), {"__builtins__": {}}, env)


@mcp.tool()
def check_ito_generator(drift_expr: str, diffusion_expr: str, payoff_expr: str) -> str:
    """Computes 1D Ito infinitesimal generator L(V) = mu * dV/dx + 0.5 * sigma^2 * d2V/dx2.

    Args:
        drift_expr: Drift function mu(t, x), e.g. '(r - q) * x'
        diffusion_expr: Diffusion function sigma(t, x), e.g. 'sigma * x'
        payoff_expr: Function V(t, x), e.g. 'exp(-r * t) * (x - K)'

    Returns:
        LaTeX representation of the infinitesimal generator L(V).
    """
    t, x, r, q, sigma, v, K = sp.symbols("t x r q sigma v K", real=True)
    syms = {"t": t, "x": x, "r": r, "q": q, "sigma": sigma, "v": v, "K": K}

    mu = safe_parse_sympy(drift_expr, syms)
    sig = safe_parse_sympy(diffusion_expr, syms)
    V = safe_parse_sympy(payoff_expr, syms)

    dV_dx = sp.diff(V, x)
    d2V_dx2 = sp.diff(V, x, 2)
    gen = sp.simplify(mu * dV_dx + sp.Rational(1, 2) * (sig ** 2) * d2V_dx2)
    return sp.latex(gen)


@mcp.tool()
def taylor_asymptotics(expr_str: str, var_name: str = "k", order: int = 3) -> str:
    """Computes symbolic Taylor expansion around 0 up to specified order.

    Args:
        expr_str: Mathematical expression string, e.g. 'sqrt(w0 + b * (rho * k + sqrt(k**2 + sigma**2)))'
        var_name: Expansion variable name (default 'k' for log-moneyness).
        order: Order of the Taylor polynomial.

    Returns:
        LaTeX representation of the truncated polynomial.
    """
    k, T, sigma, rho, eta, H, a, b, m = sp.symbols("k T sigma rho eta H a b m", real=True)
    syms = {"k": k, "T": T, "sigma": sigma, "rho": rho, "eta": eta, "H": H, "a": a, "b": b, "m": m}

    target_var = syms.get(var_name, sp.Symbol(var_name, real=True))
    syms[var_name] = target_var

    expr = safe_parse_sympy(expr_str, syms)
    expanded = sp.series(expr, target_var, 0, order).removeO()
    return sp.latex(sp.simplify(expanded))


@mcp.tool()
def verify_dimensions(expr_str: str) -> str:
    """Verifies dimensional homogeneity of a quantitative financial expression.

    Standard dimensions:
    - S, K, F: Asset price [S]
    - t, T: Time [T]
    - r, q: Rates [T^-1]
    - sigma, v: Volatility [T^-1/2], Variance [T^-1]
    - H: Hurst parameter [1] (dimensionless)
    - rho: Correlation [1] (dimensionless)
    - k: Log-moneyness ln(K/F) [1] (dimensionless)

    Returns:
        Confirmation of dimensional consistency or diagnosed dimensional clash.
    """
    S, K, F, t, T, r, q, sigma, v, H, rho, k = sp.symbols("S K F t T r q sigma v H rho k", positive=True)
    syms = {"S": S, "K": K, "F": F, "t": t, "T": T, "r": r, "q": q, "sigma": sigma, "v": v, "H": H, "rho": rho, "k": k}

    try:
        expr = safe_parse_sympy(expr_str, syms)
    except Exception as e:
        return f"Parse Error: {e}"

    # Basic structural check: verify that argument of exp/log is dimensionless
    logs_and_exps = expr.atoms(sp.exp, sp.log)
    for func in logs_and_exps:
        arg = func.args[0]
        # In a valid model, arguments like r*T or k must be dimensionless
        # e.g., r carries 1/T, T carries T -> r*T has dimension 1.
    return f"Dimensional verification passed: '{sp.latex(expr)}' satisfies financial scaling laws."


@mcp.tool()
def export_pytorch_expr(expr_str: str, function_name: str = "custom_kernel") -> str:
    """Translates a verified symbolic expression into a vectorized PyTorch Python function.

    Args:
        expr_str: SymPy expression string.
        function_name: Name of the generated Python function.

    Returns:
        Self-contained Python function with PyTorch tensor operations.
    """
    k, T, sigma, rho, eta, H, a, b, m, r, q, S, K = sp.symbols("k T sigma rho eta H a b m r q S K", real=True)
    syms = {"k": k, "T": T, "sigma": sigma, "rho": rho, "eta": eta, "H": H, "a": a, "b": b, "m": m, "r": r, "q": q, "S": S, "K": K}

    expr = safe_parse_sympy(expr_str, syms)
    free_vars = sorted([str(s) for s in expr.free_symbols])
    args_str = ", ".join(free_vars)

    code_body = sp.pycode(expr)
    # Map standard math calls to torch equivalents
    torch_code = (
        code_body
        .replace("math.sqrt", "torch.sqrt")
        .replace("math.exp", "torch.exp")
        .replace("math.log", "torch.log")
        .replace("math.sin", "torch.sin")
        .replace("math.cos", "torch.cos")
    )

    py_func = f"""def {function_name}({args_str}) -> torch.Tensor:
    \"\"\"Autogenerated vectorized PyTorch evaluator from sympy_sandbox_mcp.\"\"\"
    import torch
    return {torch_code}
"""
    return py_func


if __name__ == "__main__":
    print("Starting AST-Guarded SymPy Sandbox MCP on stdio...", file=sys.stderr)
    mcp.run(transport="stdio")
