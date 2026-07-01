# mcp_verifier_server.py
# Stage 4: Symbolic verifier exposed as a standalone MCP server.
# Transport: stdio (subprocess, for Colab). SSE swap is a one-liner at Stage 5.

import re
import json
import sympy as sp
from sympy import I, Rational, simplify, zeros, Symbol
from sympy import sympify
from itertools import combinations
from typing import NamedTuple, Literal
from mcp.server.fastmcp import FastMCP

# ── MCP server instance ───────────────────────────────────────────────────────
mcp = FastMCP("lie-algebra-verifier")


# ── SymPy helpers (verbatim from Stage 0 / Stage 2) ──────────────────────────

def su_n_generators(N):
    """
    Generalized Gell-Mann matrices for su(N), in textbook order.
    For each new dimension k=1,...,N-1: emit sym+antisym pairs for all j<k,
    then the diagonal generator completing that dimension.
    Returns list of N×N SymPy matrices (lambda_a, NOT divided by 2).
    """
    gens = []
    for k in range(1, N):
        for j in range(k):
            sym = zeros(N, N)
            sym[j, k] = 1
            sym[k, j] = 1
            gens.append(sym)

            antisym = zeros(N, N)
            antisym[j, k] = -I
            antisym[k, j] = I
            gens.append(antisym)

        kk = k + 1
        diag_entries = [1] * (kk - 1) + [-(kk - 1)] + [0] * (N - kk)
        norm = sp.sqrt(Rational(2, kk * (kk - 1)))
        diag_mat = norm * sp.diag(*diag_entries)
        gens.append(diag_mat)
    return gens


def commutator(A, B):
    return A * B - B * A


def f_abc_computed(a, b, c, T):
    """f_abc = -2i * Tr([T_a, T_b] T_c)"""
    comm = commutator(T[a], T[b])
    val = -2 * I * (comm * T[c]).trace()
    return simplify(val)


# ── Claim parsing (verbatim from Stage 2) ────────────────────────────────────

class ParsedClaim(NamedTuple):
    tag: Literal["COMMUTATOR", "STRUCTURE_CONST"]
    N: int
    raw_expr: str


def parse_claim(text: str) -> ParsedClaim:
    match = re.search(r"^CLAIM:\s*(.+)$", text.strip(), re.MULTILINE)
    if not match:
        raise ValueError("No line starting with 'CLAIM:' found in solver output.")
    claim_line = match.group(1)
    parts = [p.strip() for p in claim_line.split("|")]
    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 pipe-separated fields, got {len(parts)}: {claim_line!r}"
        )
    algebra_str, tag, expr = parts
    algebra_match = re.fullmatch(r"su\((\d+)\)", algebra_str)
    if not algebra_match:
        raise ValueError(f"Could not parse algebra string {algebra_str!r}; expected su(N).")
    N = int(algebra_match.group(1))
    if N < 2:
        raise ValueError(f"su(N) requires N >= 2, got N={N}.")
    if tag not in ("COMMUTATOR", "STRUCTURE_CONST"):
        raise ValueError(f"Unrecognized tag {tag!r}.")
    if not expr:
        raise ValueError("Expression field is empty.")
    return ParsedClaim(tag=tag, N=N, raw_expr=expr)


def parse_commutator_expr(raw_expr: str, T: list) -> tuple:
    sides = raw_expr.split("=")
    if len(sides) != 2:
        raise ValueError(f"Expected exactly one '=' in expression: {raw_expr!r}")
    lhs_str, rhs_str = sides[0].strip(), sides[1].strip()

    lhs_match = re.fullmatch(r"\[\s*T_?(\d+)\s*,\s*T_?(\d+)\s*\]", lhs_str)
    if not lhs_match:
        raise ValueError(f"Could not parse LHS as [Ta, Tb]: {lhs_str!r}")
    a, b = int(lhs_match.group(1)), int(lhs_match.group(2))

    n_gens = len(T)
    if not (1 <= a <= n_gens) or not (1 <= b <= n_gens):
        raise ValueError(f"LHS indices T{a}, T{b} out of range for {n_gens} generators.")

    lhs_matrix = T[a - 1] * T[b - 1] - T[b - 1] * T[a - 1]

    rhs_indices = sorted(set(int(m) for m in re.findall(r"T_?(\d+)", rhs_str)))
    if not rhs_indices:
        raise ValueError(f"No T_k terms found on RHS: {rhs_str!r}")
    for k in rhs_indices:
        if not (1 <= k <= n_gens):
            raise ValueError(f"RHS index T{k} out of range for {n_gens} generators.")

    placeholder_syms = {k: Symbol(f"__T{k}__") for k in rhs_indices}
    rhs_substituted = re.sub(
        r"T_?(\d+)",
        lambda m: f"__T{int(m.group(1))}__",
        rhs_str,
    )
    rhs_expr = sympify(rhs_substituted, locals={"i": I})

    rhs_matrix = zeros(T[0].shape[0], T[0].shape[1])
    for k, sym in placeholder_syms.items():
        coeff = rhs_expr.coeff(sym)
        rhs_matrix += coeff * T[k - 1]

    reconstructed = sum(rhs_expr.coeff(sym) * sym for sym in placeholder_syms.values())
    leftover = sympify(rhs_expr - reconstructed)
    if leftover != 0:
        raise ValueError(f"RHS has unexplained leftover after coefficient extraction: {leftover}")

    return lhs_matrix, rhs_matrix


def parse_structure_const_expr(raw_expr: str, T: list) -> tuple:
    sides = raw_expr.split("=")
    if len(sides) != 2:
        raise ValueError(f"Expected exactly one '=' in expression: {raw_expr!r}")
    lhs_str, rhs_str = sides[0].strip(), sides[1].strip()

    lhs_match = re.fullmatch(r"f_?(\d)_?(\d)_?(\d)", lhs_str)
    if not lhs_match:
        raise ValueError(f"Could not parse LHS as f_abc: {lhs_str!r}")
    a1, b1, c1 = (int(d) for d in lhs_match.groups())

    n_gens = len(T)
    for idx in (a1, b1, c1):
        if not (1 <= idx <= n_gens):
            raise ValueError(f"Index {idx} out of range for {n_gens} generators.")
    if len({a1, b1, c1}) != 3:
        raise ValueError(f"f_abc indices must be distinct, got ({a1},{b1},{c1}).")

    claimed_value = sympify(rhs_str)
    computed_value = f_abc_computed(a1 - 1, b1 - 1, c1 - 1, T)
    return computed_value, claimed_value


# ── The MCP tool ──────────────────────────────────────────────────────────────

@mcp.tool()
def verify_lie_algebra_identity(claim_text: str) -> str:
    """
    Symbolically verify a Lie algebra identity claim emitted by the Solver.

    Accepts the full solver output text (the CLAIM line is extracted from it).
    Returns a JSON string with fields:
      verified (bool | null), claim_tag (str | null),
      claim_algebra_N (int | null), detail (str)
    """
    try:
        claim = parse_claim(claim_text)
    except ValueError as e:
        return json.dumps({
            "verified": None, "claim_tag": None,
            "claim_algebra_N": None, "detail": f"UNPARSEABLE: {e}"
        })

    try:
        gens = su_n_generators(claim.N)
        T_local = [g / 2 for g in gens]

        if claim.tag == "COMMUTATOR":
            lhs, rhs = parse_commutator_expr(claim.raw_expr, T_local)
            verified = bool(simplify(lhs - rhs) == zeros(claim.N, claim.N))
            detail = f"Checked [Ta,Tb] vs claimed RHS for su({claim.N}). Equal: {verified}"
        else:
            computed, claimed_val = parse_structure_const_expr(claim.raw_expr, T_local)
            verified = bool(simplify(computed - claimed_val) == 0)
            detail = (
                f"f_abc computed={computed}, claimed={claimed_val} "
                f"for su({claim.N}). Equal: {verified}"
            )
    except ValueError as e:
        return json.dumps({
            "verified": None, "claim_tag": claim.tag,
            "claim_algebra_N": claim.N, "detail": f"PARSE_ERROR: {e}"
        })

    return json.dumps({
        "verified": verified, "claim_tag": claim.tag,
        "claim_algebra_N": claim.N, "detail": detail
    })


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
