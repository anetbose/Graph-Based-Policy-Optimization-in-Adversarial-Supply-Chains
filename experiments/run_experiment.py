"""Compare inspection policies against a best-responding smuggling network.

Examples
    python experiments/run_experiment.py --graph toy --inspections 1
    python experiments/run_experiment.py --graph random --inspections 3 --mode realized
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stackelberg import (  # noqa: E402
    GameConfig, cem_search, degree_policy, evaluate_policy, oracle_policy,
    random_supply_chain, toy_graph, uniform_policy,
)
from stackelberg.graph import ROLE_NAMES  # noqa: E402
from stackelberg.leader import linear_policy  # noqa: E402

ROLE_COLORS = ["#4C72B0", "#BBBBBB", "#DD8452", "#8FBF8F", "#C44E52"]


def draw(g, result, title, path):
    G = g.to_networkx()
    for i in G.nodes:
        G.nodes[i]["layer"] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}[int(g.roles[i])]
    pos = nx.multipartite_layout(G, subset_key="layer")
    br, cov = result["best_response"], result["coverage"]
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=[ROLE_COLORS[r] for r in g.roles],
        node_size=300 + 1500 * cov, edgecolors="black",
        linewidths=[3 if c > 0.5 else 1 for c in cov])
    nx.draw_networkx_labels(G, pos, {i: g.names[i] for i in G.nodes}, ax=ax, font_size=8)
    edgelist = [tuple(e) for e in g.edges]
    widths = 0.5 + 3 * br.flow / max(br.flow.max(), 1e-9)
    colors = ["#C44E52" if abs(d) > 1e-6 else "#555555" for d in br.delta]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=widths,
                           edge_color=colors, ax=ax, arrowsize=12)
    for r, name in ROLE_NAMES.items():
        ax.scatter([], [], c=ROLE_COLORS[r], label=name, edgecolors="black")
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.set_title(title + "\nnode size = coverage, edge width = flow, red edge = perturbed",
                 fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["toy", "random"], default="toy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspections", type=int, default=1)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--mode", choices=["expected", "realized"], default="expected")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    g = toy_graph() if args.graph == "toy" else random_supply_chain(seed=args.seed)
    cfg = GameConfig(n_inspections=args.inspections, budget=args.budget,
                     gamma=args.gamma, kappa=args.kappa, lam=args.lam,
                     mode=args.mode, seed=args.seed)
    out = Path(args.out) / f"{args.graph}_{args.mode}_N{args.inspections}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Graph: {g.n} nodes, {g.m} edges, {len(g.shells)} shells, "
          f"{len(g.illegal_buyers)} illegal buyers")
    no_inspect = evaluate_policy(
        g, uniform_policy(g), GameConfig(**{**cfg.__dict__, "n_inspections": 0}))
    print(f"Throughput with no inspections: {no_inspect['throughput']:.2f}")

    theta, history = cem_search(g, cfg, iters=args.iters, pop=args.pop, seed=args.seed)
    policies = {
        "uniform": uniform_policy(g),
        "volume-weighted": degree_policy(g),
        "learned (CEM)": linear_policy(theta, g.X),
        "oracle (uses labels)": oracle_policy(g),
    }

    rows = {}
    print(f"\n{'policy':<22}{'throughput':>12}{'defender U':>12}{'attacker U':>12}")
    for name, p in policies.items():
        r = evaluate_policy(g, p, cfg)
        rows[name] = {k: float(r[k]) for k in
                      ("throughput", "defender_utility", "attacker_utility")}
        print(f"{name:<22}{r['throughput']:>12.2f}"
              f"{r['defender_utility']:>12.2f}{r['attacker_utility']:>12.2f}")
        slug = name.split()[0].replace("-", "_")
        draw(g, r, f"{name} policy", out / f"graph_{slug}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(range(len(rows)), [v["throughput"] for v in rows.values()],
                color=["#BBBBBB", "#8FBF8F", "#4C72B0", "#C44E52"])
    axes[0].axhline(no_inspect["throughput"], ls="--", c="k", lw=1, label="no inspections")
    axes[0].set_xticks(range(len(rows)), list(rows), rotation=15, fontsize=8)
    axes[0].set_ylabel("expected illicit throughput")
    axes[0].legend(fontsize=8)
    axes[1].plot(history)
    axes[1].set_xlabel("CEM iteration")
    axes[1].set_ylabel("best defender utility")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=150)

    (out / "results.json").write_text(json.dumps(
        {"config": {k: v for k, v in vars(args).items()},
         "no_inspection_throughput": no_inspect["throughput"],
         "policies": rows, "theta": theta.tolist()}, indent=2))
    print(f"\nSaved figures and results to {out}/")


if __name__ == "__main__":
    main()
