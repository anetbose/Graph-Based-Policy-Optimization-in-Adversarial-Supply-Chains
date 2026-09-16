"""Supply-chain graphs G = (V, E, X, W, Y) from Section 2 of the paper."""

from dataclasses import dataclass

import networkx as nx
import numpy as np

# Node roles y_v (Section 2).
SUPPLIER, NORMAL_SELLER, SHELL, NORMAL_BUYER, ILLEGAL_BUYER = range(5)
ROLE_NAMES = {
    SUPPLIER: "supplier",
    NORMAL_SELLER: "normal seller",
    SHELL: "shell company",
    NORMAL_BUYER: "normal buyer",
    ILLEGAL_BUYER: "illegal buyer",
}


@dataclass
class SupplyChainGraph:
    """Directed weighted supply-chain graph.

    names   : node names, length n
    roles   : (n,) int array of roles y_v
    edges   : (m, 2) int array of directed edges (u, v)
    weights : (m,) nonnegative edge weights W(u, v)
    X       : (n, d) node feature matrix
    """

    names: list
    roles: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    X: np.ndarray

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return len(self.edges)

    def nodes_with_role(self, role: int) -> np.ndarray:
        return np.flatnonzero(self.roles == role)

    @property
    def suppliers(self) -> np.ndarray:  # S
        return self.nodes_with_role(SUPPLIER)

    @property
    def shells(self) -> np.ndarray:  # M
        return self.nodes_with_role(SHELL)

    @property
    def illegal_buyers(self) -> np.ndarray:  # I
        return self.nodes_with_role(ILLEGAL_BUYER)

    def attackable_edges(self) -> np.ndarray:
        """Boolean mask for E~, the edges touching at least one shell company (eq. 4)."""
        return np.isin(self.edges, self.shells).any(axis=1)

    def adjacency(self, weights=None) -> np.ndarray:
        """Weighted adjacency matrix A with A[u, v] = W(u, v)."""
        w = self.weights if weights is None else weights
        A = np.zeros((self.n, self.n))
        A[self.edges[:, 0], self.edges[:, 1]] = w
        return A

    def to_networkx(self, weights=None) -> nx.DiGraph:
        w = self.weights if weights is None else weights
        G = nx.DiGraph()
        for i, name in enumerate(self.names):
            G.add_node(i, name=name, role=int(self.roles[i]))
        for (u, v), wk in zip(self.edges, w):
            G.add_edge(int(u), int(v), weight=float(wk))
        return G


def structural_features(n: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """In/out degree and in/out weighted degree, standardized per column."""
    u, v = edges[:, 0], edges[:, 1]
    feats = np.stack([
        np.bincount(v, minlength=n),
        np.bincount(u, minlength=n),
        np.bincount(v, weights=weights, minlength=n),
        np.bincount(u, weights=weights, minlength=n),
    ], axis=1).astype(float)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    return (feats - feats.mean(axis=0)) / std


def build_features(n, edges, weights, roles, include_roles=False,
                   noise=0.0, rng=None) -> np.ndarray:
    """Node features X.

    Roles are excluded by default. If the defender could read Y directly, the
    best policy would be to inspect the labelled shells and illegal buyers,
    which makes the game trivial.
    """
    X = structural_features(n, edges, weights)
    if noise > 0:
        rng = np.random.default_rng() if rng is None else rng
        X = X + noise * rng.standard_normal(X.shape)
    if include_roles:
        X = np.hstack([X, np.eye(5)[roles]])
    return X


def _make(names, roles, edge_list, include_roles=False, noise=0.0, rng=None):
    edges = np.array([(u, v) for u, v, _ in edge_list], dtype=int)
    weights = np.array([w for _, _, w in edge_list], dtype=float)
    roles = np.asarray(roles, dtype=int)
    X = build_features(len(names), edges, weights, roles, include_roles, noise, rng)
    return SupplyChainGraph(list(names), roles, edges, weights, X)


def toy_graph(include_roles: bool = False) -> SupplyChainGraph:
    """Small graph in the spirit of Figure 1.

    S supplies two authorized sellers A and B. Two shell companies C and D
    forward chips to the illegal buyer I. E is a normal buyer.
    """
    names = ["S", "A", "B", "C", "D", "I", "E"]
    roles = [SUPPLIER, NORMAL_SELLER, NORMAL_SELLER, SHELL, SHELL,
             ILLEGAL_BUYER, NORMAL_BUYER]
    S, A, B, C, D, I, E = range(7)
    edge_list = [
        (S, A, 5), (S, B, 6),
        (A, C, 6), (A, D, 2), (B, D, 4), (B, C, 1), (B, E, 3),
        (C, I, 5), (D, I, 5),
    ]
    return _make(names, roles, edge_list, include_roles)


def random_supply_chain(n_suppliers=3, n_sellers=8, n_shells=4, n_buyers=6,
                        n_illegal=3, p_edge=0.35, p_leak=0.25,
                        max_weight=10, include_roles=False, noise=0.0,
                        seed=None) -> SupplyChainGraph:
    """Layered random graph: suppliers -> sellers -> (buyers | shells -> illegal buyers).

    Shell companies buy from authorized sellers, matching the diversion
    pattern described in Section 1. Sellers also sell directly to illegal
    buyers with a small probability.
    """
    rng = np.random.default_rng(seed)
    counts = [n_suppliers, n_sellers, n_shells, n_buyers, n_illegal]
    layer_roles = [SUPPLIER, NORMAL_SELLER, SHELL, NORMAL_BUYER, ILLEGAL_BUYER]
    prefixes = ["S", "N", "M", "B", "I"]
    names, roles, layers, idx = [], [], {}, 0
    for c, r, p in zip(counts, layer_roles, prefixes):
        layers[r] = list(range(idx, idx + c))
        names += [f"{p}{k}" for k in range(c)]
        roles += [r] * c
        idx += c

    def w():
        return int(rng.integers(1, max_weight + 1))

    edges = {}

    def connect(srcs, dsts, p, ensure_in=False):
        for d in dsts:
            chosen = [s for s in srcs if rng.random() < p]
            if ensure_in and not chosen:
                chosen = [int(rng.choice(srcs))]
            for s in chosen:
                edges[(s, d)] = w()

    connect(layers[SUPPLIER], layers[NORMAL_SELLER], p_edge, ensure_in=True)
    connect(layers[NORMAL_SELLER], layers[NORMAL_BUYER], p_edge, ensure_in=True)
    connect(layers[NORMAL_SELLER], layers[SHELL], p_edge, ensure_in=True)
    connect(layers[SHELL], layers[ILLEGAL_BUYER], p_edge, ensure_in=True)
    connect(layers[NORMAL_SELLER], layers[ILLEGAL_BUYER], p_leak * 0.3)
    # every shell needs an outlet to at least one illegal buyer
    for s in layers[SHELL]:
        if not any(u == s for u, _ in edges):
            edges[(s, int(rng.choice(layers[ILLEGAL_BUYER])))] = w()

    edge_list = [(u, v, wt) for (u, v), wt in sorted(edges.items())]
    return _make(names, roles, edge_list, include_roles, noise, rng)
