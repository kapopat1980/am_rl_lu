# AM-RL-LU: Adaptive Movement-Based RL Location Update

Simulation code accompanying the manuscript *"AM-RL-LU: An Adaptive
Movement-Based Reinforcement Learning Strategy for Location Update in
Cellular Networks"*.

## What this is

A custom, dependency-light (numpy/pandas/matplotlib only — no PyTorch)
discrete-event simulator for cellular location-update (LU) strategies,
comparing classical static schemes (movement-based, distance-based,
time-based) against a reinforcement-learning-adaptive scheme (AM-RL-LU),
under both stationary and non-stationary (regime-switching) mobility/call
conditions.

## Repository structure

```
am_rl_lu/
├── mobility/
│   ├── models.py       # Random Walk, Fluid Flow, Gauss-Markov mobility models
│   └── cellgrid.py     # Hexagonal cell grid (axial coordinates)
├── cost/
│   └── cost_model.py   # LU + paging signaling cost model
├── agents/
│   ├── q_learning.py         # Tabular Q-learning agent
│   ├── dqn_numpy.py          # Dependency-free DQN (Adam + Double DQN), numpy only
│   └── analytic_controller.py  # Closed-form adaptive threshold controller (recommended)
├── experiments/
│   ├── environment.py            # Core simulation engine (shared by all schemes)
│   ├── baselines.py               # Grid-search optimizers for static schemes
│   ├── nonstationary.py           # Regime-switching trace generation
│   ├── run_experiment.py          # Stationary-regime experiment (sanity check)
│   └── run_nonstationary_experiment.py  # Main experiment (paper's headline result)
└── results/
    ├── raw_results.csv            # Stationary-regime results
    ├── final_comprehensive_results.csv  # Non-stationary results, all variants (main)
    ├── nonstationary_comparison.png
    └── adaptation_trajectory.png

## Which variant should I use?

Start with `agents/analytic_controller.py` (paper Section 4.3). It requires
no training, outperforms both trained-RL variants in the paper's evaluation,
and is the simplest to deploy or extend. The Q-learning and DQN agents are
included for comparison and as a worked example of where a naive trained-RL
approach can go wrong (see the paper's Section 6.2-6.3 diagnosis) if you want
to explore that direction further.
```

## Reproducing the paper's results

```bash
pip install numpy pandas matplotlib

# Stationary-regime sanity check (Section 6.4 of the paper)
python experiments/run_experiment.py

# Main non-stationary evaluation (Section 6.1-6.3, Table 1, Figures 1-2)
python experiments/run_nonstationary_experiment.py
```

Both scripts write their results to `results/`. The non-stationary script
trains all four AM-RL-LU variants (Q-learning / DQN × stationary-trained /
switching-trained) and evaluates all ten schemes (4 AM-RL-LU + 3 static ×
2 tuning regimes) on a held-out test set (seed base 24680, offset +900000)
that is disjoint from the validation seeds used during hyperparameter
selection (seed base 999, offset +500000) and from all training seeds.
This separation is intentional — see Section 5.3-5.4 of the paper — and
should be preserved in any modification of this code, to avoid overfitting
reported numbers to the evaluation set.

## Key modeling notes (see paper Section 3 for full derivations)

- **Cell radius** is calibrated to 6.0 units so that mobiles cross cell
  boundaries at a realistic rate (~0.14-0.18 crossings/step) relative to
  their mobility model's speed. Do not reduce this without re-checking
  crossing rates — a too-small radius causes near-every-step crossings,
  which compresses the effective range of any movement/distance threshold.
- **Time-based scheme's paging radius is derived, not free**: it equals
  `ceil(v_max * period / adjacent_cell_distance)`. Treating it as an
  independently tunable parameter (as an earlier version of this code did)
  lets a grid search pick an unrealistically small radius regardless of
  update frequency — see `time_based_paging_radius()` in
  `experiments/environment.py` and Section 3.3 of the paper.
- **Static baselines are evaluated two ways**: "oracle full-trace" (grid
  search sees the entire trace, a theoretical upper bound) and "realistic
  pre-switch-tuned" (grid search sees only the pre-shift segment, what an
  operator could actually configure). The paper's headline comparison is
  against the realistic baseline — see Section 5.2.

## License / citation

If you use this code, please cite the accompanying paper (see repository
root for citation details once published).
