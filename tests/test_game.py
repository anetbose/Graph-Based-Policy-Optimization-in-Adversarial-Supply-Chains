import networkx as nx
import numpy as np

from stackelberg import (
    GameConfig, best_response, estimate_coverage, evaluate_policy,
    oracle_policy, random_supply_chain, toy_graph, uniform_policy,
)


def nx_max_flow(g):
    G = g.to_networkx()
    for (u, v), w in zip(g.edges, g.weights):
        G[u][v]["capacity"] = w
    for s in g.suppliers:
        G.add_edge("src", int(s), capacity=float("inf"))
    for i in g.illegal_buyers:
        G.add_edge(int(i), "snk", capacity=float("inf"))
    return nx.maximum_flow_value(G, "src", "snk")


def test_zero_budget_matches_max_flow():
    g = toy_graph()
    br = best_response(g, np.zeros(g.n), budget=0.0, gamma=0.0)
    assert np.isclose(br.throughput, nx_max_flow(g))
    assert np.isclose(br.throughput, 10.0)


def test_coverage_sums_to_n():
    g = random_supply_chain(seed=1)
    for N in (1, 3, 5):
        c = estimate_coverage(uniform_policy(g), N, n_samples=500)
        assert np.isclose(c.sum(), N)
        assert np.all((c >= 0) & (c <= 1))


def test_budget_never_hurts_attacker():
    g = random_supply_chain(seed=2)
    cov = estimate_coverage(uniform_policy(g), 2)
    vals = [best_response(g, cov, budget=b, gamma=3.0).throughput for b in (0, 2, 5, 10)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_perturbation_respects_constraints():
    g = random_supply_chain(seed=3)
    B, G = 5.0, 2.0
    br = best_response(g, np.zeros(g.n), budget=B, gamma=G)
    assert np.abs(br.delta).sum() <= B + 1e-7
    assert np.all(np.abs(br.delta) <= G + 1e-7)
    assert np.all(br.delta[~g.attackable_edges()] == 0)
    assert np.all(br.flow <= g.weights + br.delta + 1e-7)


def test_inspecting_sink_blocks_all_flow_in_realized_mode():
    g = toy_graph()
    I = g.illegal_buyers
    br = best_response(g, np.zeros(g.n), 4.0, 3.0, blocked_nodes=I)
    assert np.isclose(br.throughput, 0.0)


def test_oracle_beats_uniform():
    g = random_supply_chain(seed=4)
    cfg = GameConfig(n_inspections=3, budget=5.0, gamma=3.0)
    u = evaluate_policy(g, uniform_policy(g), cfg)["defender_utility"]
    o = evaluate_policy(g, oracle_policy(g), cfg)["defender_utility"]
    assert o >= u
