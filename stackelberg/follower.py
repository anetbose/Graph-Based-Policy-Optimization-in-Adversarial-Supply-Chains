"""Follower (smuggling network) best response, eqs. (4)-(9) and (11).

For a fixed coverage c, the follower's problem

    max_{delta in D(W)}  max_{f in F(W + delta)}  sum_i (1 - c(i)) F_i(f) - kappa * sum |delta|

is a single linear program, because the perturbed capacity W + delta enters
the flow constraints linearly. Writing delta = d_plus - d_minus with
d_plus, d_minus >= 0 turns |delta| into d_plus + d_minus, so the best response
is exact rather than approximate.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass
class AttackResult:
    utility: float        # U_A, eq. (9)
    throughput: float     # sum_i (1 - c(i)) F_i(f), eq. (8)
    flow: np.ndarray      # (m,) optimal flow f
    delta: np.ndarray     # (m,) optimal perturbation
    sink_inflow: np.ndarray  # (n,) F_i(f), nonzero only at illegal buyers


def best_response(g, coverage, budget, gamma, kappa=0.0, blocked_nodes=None):
    """Solve the follower LP.

    g             : SupplyChainGraph
    coverage      : (n,) inclusion probabilities c(v). Only entries for
                    illegal buyers enter the objective, following eq. (8).
    budget        : B, total perturbation budget
    gamma         : Gamma, per-edge perturbation cap
    kappa         : manipulation-effort penalty
    blocked_nodes : optional node indices whose incident edges carry no flow.
                    Used for the realized-inspection variant, where the
                    follower sees the inspection set before routing.
    """
    n, m = g.n, g.m
    u, v = g.edges[:, 0], g.edges[:, 1]
    W = g.weights
    sources, sinks = g.suppliers, g.illegal_buyers

    sink_weight = np.zeros(n)
    sink_weight[sinks] = 1.0 - np.asarray(coverage)[sinks]

    # Variables x = [f (m), d_plus (m), d_minus (m)]; linprog minimizes.
    c = np.concatenate([-sink_weight[v], np.full(m, kappa), np.full(m, kappa)])

    # Flow may not leave an illegal buyer or enter a supplier, so terminals
    # act only as sinks and sources.
    f_zero = np.isin(u, sinks) | np.isin(v, sources)
    if blocked_nodes is not None and len(blocked_nodes):
        f_zero |= np.isin(u, blocked_nodes) | np.isin(v, blocked_nodes)
    attackable = g.attackable_edges()
    bounds = (
        [(0.0, 0.0) if z else (0.0, None) for z in f_zero]
        + [(0.0, gamma) if a else (0.0, 0.0) for a in attackable] * 2
    )

    I_m = sparse.identity(m, format="csr")
    Z_m = sparse.csr_matrix((m, m))
    ones = np.ones((1, m))
    A_ub = sparse.vstack([
        sparse.hstack([I_m, -I_m, I_m]),       # f <= W + delta
        sparse.hstack([Z_m, -I_m, I_m]),       # W + delta >= 0
        sparse.hstack([sparse.csr_matrix((1, m)), ones, ones]),  # sum |delta| <= B
    ], format="csr")
    b_ub = np.concatenate([W, W, [budget]])

    # Flow conservation at every node outside S and I (eq. 5).
    inc = sparse.csr_matrix(
        (np.concatenate([np.ones(m), -np.ones(m)]),
         (np.concatenate([v, u]), np.concatenate([np.arange(m)] * 2))),
        shape=(n, m),
    )
    interior = np.setdiff1d(np.arange(n), np.concatenate([sources, sinks]))
    A_eq = b_eq = None
    if len(interior):
        A_eq = sparse.hstack([inc[interior], sparse.csr_matrix((len(interior), 2 * m))])
        b_eq = np.zeros(len(interior))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"Follower LP failed: {res.message}")

    f, dp, dm = res.x[:m], res.x[m:2 * m], res.x[2 * m:]
    inflow = np.zeros(n)
    np.add.at(inflow, v, f)
    mask = np.zeros(n, bool)
    mask[sinks] = True
    inflow[~mask] = 0.0
    throughput = float(sink_weight[v] @ f)
    return AttackResult(
        utility=throughput - kappa * float(np.sum(dp + dm)),
        throughput=throughput,
        flow=f,
        delta=dp - dm,
        sink_inflow=inflow,
    )
