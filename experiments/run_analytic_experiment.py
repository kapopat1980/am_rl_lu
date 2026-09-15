"""
Produces the paper's headline results: trains all four trained-RL variants
(Q-learning / DQN x stationary-trained / switching-trained), evaluates the
analytic controller (agents/analytic_controller.py) and all six static
baselines (movement/distance/time x oracle/realistic), on the same held-out
test set, and writes results/final_comprehensive_results.csv -- the exact
file Tables 1-2 and Figures 5-6 in the paper are built from.

This is the primary reproduction script for the paper. run_experiment.py and
run_nonstationary_experiment.py remain useful as earlier/simpler sanity
checks (Section VI-E and the trained-RL-only comparison respectively), but
this script is the one to run to reproduce the paper's actual reported
numbers end to end.

Usage:
    python experiments/run_analytic_experiment.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from experiments.run_nonstationary_experiment import (
    make_switching_trace, cost_model, EPOCH_LEN,
    realistic_fixed_threshold, realistic_fixed_time,
    oracle_fixed_threshold, oracle_fixed_time,
    train_agent_switching, train_agent_stationary_only, eval_agent_on_trace,
    V_MAX_ASSUMED, ADJACENT_CELL_DIST,
)
from experiments.environment import AdaptiveLUEnv, run_static_threshold, run_time_based
from agents.analytic_controller import analytic_best_threshold
from agents.q_learning import QLearningAgent
from agents.dqn_numpy import DQNAgent

# Analytic controller hyperparameters (Section IV-C / V-D of the paper)
SMOOTHING = 0.85
MAX_STEP = 5

# Held-out test set: disjoint from the validation seeds used during all
# hyperparameter/model selection (seed base 999, offset +500000) -- see
# Section V-C of the paper.
TEST_SEED_BASE = 24680
TEST_SEED_OFFSET = 900000
N_TEST_TRACES = 30


def run_analytic(trace, smoothing=SMOOTHING, max_step=MAX_STEP):
    env = AdaptiveLUEnv(trace, cost_model, epoch_len=EPOCH_LEN, state_smoothing=smoothing)
    obs = env.reset()
    done = False
    total = 0.0
    while not done:
        cr, cl = obs[0], obs[1]
        target = analytic_best_threshold(cr, cl, EPOCH_LEN, cost_model)
        action = int(np.clip(target - env.threshold, -max_step, max_step))
        obs, r, done, info = env.step(action)
        total += info["total_cost"]
    return total


def main():
    print("Training Q-learning (n_bins=5, 300 episodes) and DQN (400 episodes) variants...")
    q_switch = train_agent_switching(QLearningAgent, {"seed": 42, "n_bins": 5}, episodes=300)
    q_stat = train_agent_stationary_only(QLearningAgent, {"seed": 42, "n_bins": 5}, episodes=300)
    dqn_switch = train_agent_switching(DQNAgent, {"seed": 42})
    dqn_stat = train_agent_stationary_only(DQNAgent, {"seed": 42})

    print(f"Evaluating on held-out test set (seed base {TEST_SEED_BASE}, offset +{TEST_SEED_OFFSET})...")
    eval_seed_rng = np.random.default_rng(TEST_SEED_BASE)
    rows = []
    for i in range(N_TEST_TRACES):
        seed = int(eval_seed_rng.integers(0, 10 ** 6)) + TEST_SEED_OFFSET
        trace, segments = make_switching_trace(seed)

        realistic_mv = realistic_fixed_threshold(segments, seed, "movement")
        realistic_mv_cost = run_static_threshold(trace, cost_model, realistic_mv, mode="movement")["total_cost"]
        realistic_ds = realistic_fixed_threshold(segments, seed, "distance")
        realistic_ds_cost = run_static_threshold(trace, cost_model, realistic_ds, mode="distance")["total_cost"]
        realistic_tm = realistic_fixed_time(segments, seed)
        realistic_tm_cost = run_time_based(trace, cost_model, realistic_tm, V_MAX_ASSUMED, ADJACENT_CELL_DIST)["total_cost"]

        oracle_mv_cost = oracle_fixed_threshold(trace, "movement")
        oracle_ds_cost = oracle_fixed_threshold(trace, "distance")
        oracle_tm_cost = oracle_fixed_time(trace)

        analytic_cost = run_analytic(trace)
        q_switch_cost = eval_agent_on_trace(q_switch, trace)
        q_stat_cost = eval_agent_on_trace(q_stat, trace)
        dqn_switch_cost = eval_agent_on_trace(dqn_switch, trace)
        dqn_stat_cost = eval_agent_on_trace(dqn_stat, trace)

        for scheme, cost in [
            ("Static Movement (tuned on pre-switch regime)", realistic_mv_cost),
            ("Static Distance (tuned on pre-switch regime)", realistic_ds_cost),
            ("Static Time (tuned on pre-switch regime)", realistic_tm_cost),
            ("Static Movement (oracle, theoretical upper bound)", oracle_mv_cost),
            ("Static Distance (oracle, theoretical upper bound)", oracle_ds_cost),
            ("Static Time (oracle, theoretical upper bound)", oracle_tm_cost),
            ("AM-RL-LU Analytic (proposed)", analytic_cost),
            ("AM-RL-LU Q-learning (switching-trained)", q_switch_cost),
            ("AM-RL-LU Q-learning (stationary-trained)", q_stat_cost),
            ("AM-RL-LU DQN (switching-trained)", dqn_switch_cost),
            ("AM-RL-LU DQN (stationary-trained)", dqn_stat_cost),
        ]:
            rows.append({"eval_trace": i, "seed": seed, "scheme": scheme, "total_cost": cost})

        if (i + 1) % 5 == 0:
            print(f"  evaluated {i + 1}/{N_TEST_TRACES} test traces")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "final_comprehensive_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}\n")
    print(df.groupby("scheme")["total_cost"].mean().sort_values())
    return df


if __name__ == "__main__":
    main()
