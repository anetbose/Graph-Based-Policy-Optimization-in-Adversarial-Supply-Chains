# Graph-Based Stackelberg Model for Adversarial Supply Chain Enforcement

Ananya Bose

*The PDF uses the NeurIPS 2025 template, which is why it lists anonymous authors and a submission notice. It is not currently under review at any venue.*

## Overview

Export controls on advanced AI chips depend on enforcement agencies that have far fewer inspectors than the global semiconductor trade has intermediaries, while smuggling networks can replace shell companies and reroute shipments faster than agencies can respond. The paper models this as a Stackelberg security game on a directed supply-chain graph. The enforcement agency moves first by committing to a randomized policy for inspecting firms, and the smuggling network then observes that policy, perturbs trade capacities within a fixed budget, and routes chips from suppliers to illegal buyers through the paths that remain open.

This repository implements that model, computes the smuggler's best response exactly, searches for a good inspection policy, and compares the result against simple baselines on synthetic supply chains.

![Learned inspection policy on a random supply chain](docs/learned_policy.png)

Node size shows how often each firm is inspected and edge width shows the smuggler's optimal flow. The learned policy concentrates inspections on the illegal buyer I0, so the smuggler spends its perturbation budget on the shell-company edges leading to I1 and I2, which appear in red.

## How the model maps to the code

| Paper | Code |
|---|---|
| Supply-chain graph G = (V, E, X, W, Y), Section 2 | `stackelberg/graph.py` |
| Inspection policy and coverage, eqs. (1) to (3) | `stackelberg/leader.py` |
| Attack set, feasible flows and smuggler utility, eqs. (4) to (9) and (11) | `stackelberg/follower.py` |
| Defender utility and equilibrium, eqs. (10) and (12) | `stackelberg/game.py` |
| Policy comparison and figures | `experiments/run_experiment.py` |
| Correctness checks | `tests/test_game.py` |

## Method

For a fixed inspection policy, the smuggler chooses edge perturbations δ and then routes the maximum possible flow through the perturbed graph. The perturbed capacities W + δ enter the flow constraints linearly, and writing δ as the difference of two nonnegative vectors turns the absolute values in the budget constraint into linear terms. The smuggler's full problem therefore reduces to a single linear program, which `best_response` solves exactly with the HiGHS solver.

The agency's policy is a softmax over a linear score of node features, π_θ = softmax(Xθ). Inspecting N distinct firms means sampling without replacement, so the probability that each firm is inspected is estimated by Monte Carlo, and these estimates sum to N as eq. (3) requires. The agency's objective has no useful gradient through the smuggler's linear program, so `cem_search` tunes θ with the cross-entropy method, a derivative-free search that works well at this scale.

Node features are in-degree, out-degree, and the corresponding weighted degrees. Role labels are excluded by default because an agency that could read them would simply inspect the known shells and illegal buyers. The `oracle` baseline does exactly that and serves as a reference point.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests
python experiments/run_experiment.py --graph toy --inspections 1
python experiments/run_experiment.py --graph random --inspections 3
python experiments/run_experiment.py --graph random --inspections 3 --mode realized
```

Each run prints a comparison table and saves figures and a `results.json` file to `results/`. Budget, per-edge cap, effort penalty and inspection cost can be set with `--budget`, `--gamma`, `--kappa` and `--lam`.

## Example results

On a random supply chain with 24 firms, 4 shell companies and 3 illegal buyers, the smuggler delivers 33 units with no inspections. With three inspections in `expected` mode, the policies compare as follows.

| Policy | Expected illicit throughput |
|---|---|
| Uniform | 29.09 |
| Volume-weighted | 29.44 |
| Oracle (uses true labels) | 18.74 |
| Learned (CEM) | 16.16 |

These numbers come from a single synthetic graph with fixed seeds and should be read as an illustration of the model rather than as evidence about real enforcement. The learned policy beats the oracle because, under eq. (8), inspections of shell companies do not reduce the smuggler's payoff, and the oracle spends part of its coverage on them.

![Policy comparison and search progress](docs/summary.png)

## Evaluation modes

The paper describes two timings for the game, and the code supports both. In `expected` mode the smuggler responds to the inspection probabilities, following eqs. (8) to (10). In `realized` mode the smuggler sees each sampled set of inspected firms before routing, any inspected firm carries no flow, and results are averaged over sampled sets, which matches the description in Section 2.1.

## Implementation choices

Some details are left open by the paper, and the code fixes them as follows. Flow on each edge is bounded by its perturbed capacity, f ≤ W + δ. Flow cannot leave an illegal buyer or enter a supplier. The attackable edge set Ẽ contains every edge with a shell company at either end. Normal buyers conserve flow as eq. (5) requires, so they cannot absorb illicit throughput.

## Limitations and next steps

The graphs are small and synthetic, and the cross-entropy search will not scale to networks with thousands of firms. The modelling limitations discussed in Section 4 of the paper also apply, including full observability of the network and a static graph. Planned extensions include a graph neural network policy, training across a distribution of graphs in a repeated game, and an objective that credits the agency for interdicting shell companies.

## Repository structure

```
stackelberg/        model code (graph, leader, follower, game)
experiments/        experiment script and plotting
tests/              pytest checks against networkx max-flow and model constraints
docs/               figures used in this README
```

## Citation

```bibtex
@misc{bose2026stackelberg,
  author = {Bose, Ananya},
  title  = {Graph-Based Stackelberg Model for Adversarial Supply Chain Enforcement},
  year   = {2026},
  note   = {Preprint}
}
```
