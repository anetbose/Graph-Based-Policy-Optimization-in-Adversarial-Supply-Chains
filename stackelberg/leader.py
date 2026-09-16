"""Leader (enforcement agency) policies and coverage, eqs. (1)-(3)."""

import numpy as np


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def linear_policy(theta, X):
    """pi_theta(W, X) = softmax(X theta), a point in the simplex (eq. 1)."""
    return softmax(X @ theta)


def sample_inspections(p, n_inspections, n_samples, rng):
    """Draw inspection sets of N distinct nodes, sequentially without replacement."""
    n = len(p)
    return np.stack([
        rng.choice(n, size=n_inspections, replace=False, p=p)
        for _ in range(n_samples)
    ])


def estimate_coverage(p, n_inspections, n_samples=2000, seed=0):
    """Monte Carlo estimate of c(v) = P(v in inspection set) (eq. 2).

    The estimate sums to N exactly, matching eq. (3). A fixed seed gives
    common random numbers across policies, which keeps the leader's search
    less noisy.
    """
    rng = np.random.default_rng(seed)
    samples = sample_inspections(p, n_inspections, n_samples, rng)
    counts = np.bincount(samples.ravel(), minlength=len(p))
    return counts / n_samples


def _normalize(scores, eps=1e-9):
    s = np.asarray(scores, float) + eps
    return s / s.sum()


def uniform_policy(g):
    return np.full(g.n, 1.0 / g.n)


def degree_policy(g):
    """Inspect in proportion to total traded volume through each node."""
    vol = np.zeros(g.n)
    np.add.at(vol, g.edges[:, 0], g.weights)
    np.add.at(vol, g.edges[:, 1], g.weights)
    return _normalize(vol)


def oracle_policy(g):
    """Uniform over true shells and illegal buyers. An upper-bound reference that uses Y."""
    s = np.zeros(g.n)
    s[g.shells] = 1.0
    s[g.illegal_buyers] = 1.0
    return _normalize(s)
