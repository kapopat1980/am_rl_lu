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
│   ├── run_experiment.py          # Stationary-regime experiment (sanity check, Section VI-E)
│   ├── run_nonstationary_experiment.py  # Trained-RL vs. static baselines (earlier/simpler comparison)
│   └── run_analytic_experiment.py # PRIMARY reproduction script -- produces the paper's actual
│                                   # headline results (Tables 1-2, Figures 5-6)
└── results/
    ├── raw_results.csv            # Stationary-regime results
    ├── nonstationary_results.csv  # Trained-RL vs. static baselines (from run_nonstationary_experiment.py)
    ├── final_comprehensive_results.csv  # Paper's actual headline results (from run_analytic_experiment.py)
    ├── nonstationary_comparison.png
    └── adaptation_trajectory.png
```

## Which variant should I use?

Start with `agents/analytic_controller.py` (paper Section IV-C). It requires
no training, outperforms both trained-RL variants in the paper's evaluation,
and is the simplest to deploy or extend. The Q-learning and DQN agents are
included for comparison and as a worked example of where a naive trained-RL
approach can go wrong (see the paper's Section VI-B to VI-C diagnosis) if you want
to explore that direction further.

## Reproducing the paper's results

```bash
pip install numpy pandas matplotlib

# Stationary-regime sanity check (Section VI-E of the paper)
python experiments/run_experiment.py

# Earlier/simpler comparison: trained-RL variants vs. static baselines only
python experiments/run_nonstationary_experiment.py

# PRIMARY reproduction script: produces the paper's actual reported results
# (Tables 1-2, Figures 5-6), including the analytic controller
python experiments/run_analytic_experiment.py
```

All three scripts write their results to `results/`. `run_analytic_experiment.py`
(the primary reproduction script) trains all four trained-RL variants
(Q-learning / DQN x stationary-trained / switching-trained), evaluates the
analytic controller, and evaluates all six static-baseline configurations
(movement/distance/time x oracle/realistic) -- 11 schemes in total -- on a
held-out test set (seed base 24680, offset +900000) that is disjoint from
the validation seeds used during hyperparameter selection (seed base 999,
offset +500000) and from all training seeds. `run_nonstationary_experiment.py`
performs a similar but smaller comparison (10 schemes, no analytic
controller) and is kept as an earlier, simpler point of reference. This
seed separation is intentional -- see Section V-C to V-D of the paper --
and should be preserved in any modification of this code, to avoid
overfitting reported numbers to the evaluation set.

## Key modeling notes (see paper Section III for full derivations)

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
  `experiments/environment.py` and Section III-C of the paper.
- **Static baselines are evaluated two ways**: "oracle full-trace" (grid
  search sees the entire trace, a theoretical upper bound) and "realistic
  pre-switch-tuned" (grid search sees only the pre-shift segment, what an
  operator could actually configure). The paper's headline comparison is
  against the realistic baseline — see Section V-B.

## License / citation

If you use this code, please cite the accompanying paper (see repository
root for citation details once published).
