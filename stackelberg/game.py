"""Defender utility, equilibrium search, and policy evaluation, eqs. (10)-(12)."""

from dataclasses import dataclass, field

import numpy as np

from .follower import best_response
from .leader import estimate_coverage, linear_policy, sample_inspections


@dataclass
class GameConfig:
    n_inspections: int = 1        # N
    budget: float = 4.0           # B
    gamma: float = 3.0            # Gamma
    kappa: float = 0.0            # manipulation-effort penalty
    lam: float = 0.0              # lambda, inspection-cost tradeoff
    inspection_cost: np.ndarray = field(default=None)  # C_v, defaults to ones
    mode: str = "expected"        # "expected" (eq. 8) or "realized"
    n_coverage_samples: int = 2000
    n_realized_samples: int = 64
    seed: int = 0


def evaluate_policy(g, p, cfg: GameConfig):
    """Defender and follower utilities when the leader commits to distribution p.

    expected : follower best-responds to marginal coverage, as in eqs. (8)-(10).
    realized : follower observes each sampled inspection set, inspected nodes
               carry no flow (this covers the shell interdiction described in
               Section 2.1), and utilities are averaged over samples.
    """
    cost = np.ones(g.n) if cfg.inspection_cost is None else cfg.inspection_cost
    cov = estimate_coverage(p, cfg.n_inspections, cfg.n_coverage_samples, cfg.seed)
    cost_term = cfg.lam * float(cov @ cost)

    if cfg.mode == "expected":
        br = best_response(g, cov, cfg.budget, cfg.gamma, cfg.kappa)
        throughput, u_att, last = br.throughput, br.utility, br
    elif cfg.mode == "realized":
        rng = np.random.default_rng(cfg.seed + 1)
        sets = sample_inspections(p, cfg.n_inspections, cfg.n_realized_samples, rng)
        zero = np.zeros(g.n)
        results = [best_response(g, zero, cfg.budget, cfg.gamma, cfg.kappa,
                                 blocked_nodes=s) for s in sets]
        throughput = float(np.mean([r.throughput for r in results]))
        u_att = float(np.mean([r.utility for r in results]))
        last = results[-1]
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    return {
        "defender_utility": -throughput - cost_term,   # eq. (10)
        "attacker_utility": u_att,                     # eq. (9)
        "throughput": throughput,
        "coverage": cov,
        "best_response": last,
    }


def cem_search(g, cfg: GameConfig, iters=30, pop=32, elite_frac=0.25,
               init_std=1.0, seed=0, verbose=False):
    """Approximate theta* in eq. (12) with the cross-entropy method.

    The leader objective is piecewise linear in coverage and has no useful
    gradient through the follower LP, so a derivative-free search is a
    reasonable first solver on small graphs.
    """
    rng = np.random.default_rng(seed)
    d = g.X.shape[1]
    mu, sigma = np.zeros(d), np.full(d, init_std)
    n_elite = max(1, int(pop * elite_frac))
    best_theta, best_val, history = mu.copy(), -np.inf, []

    for t in range(iters):
        thetas = mu + sigma * rng.standard_normal((pop, d))
        vals = np.array([
            evaluate_policy(g, linear_policy(th, g.X), cfg)["defender_utility"]
            for th in thetas
        ])
        elite = thetas[np.argsort(vals)[-n_elite:]]
        mu, sigma = elite.mean(axis=0), elite.std(axis=0) + 1e-3
        if vals.max() > best_val:
            best_val, best_theta = float(vals.max()), thetas[vals.argmax()].copy()
        history.append(best_val)
        if verbose:
            print(f"iter {t:3d}  best defender utility {best_val:.3f}")
    return best_theta, history
