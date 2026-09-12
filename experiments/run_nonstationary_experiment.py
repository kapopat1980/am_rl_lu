import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from mobility.cellgrid import HexCellGrid
from cost.cost_model import SignalingCostModel
from experiments.environment import LUSimTrace, AdaptiveLUEnv, run_static_threshold, run_time_based
from experiments.nonstationary import build_nonstationary_trace, sample_switching_config
from experiments.run_experiment import make_mobility, MOBILITY_KEYS, CMR_VALUES, THRESHOLD_RANGE, \
    TIME_PERIOD_RANGE, V_MAX_ASSUMED, ADJACENT_CELL_DIST
from agents.q_learning import QLearningAgent, ACTIONS
from agents.dqn_numpy import DQNAgent

N_STEPS = 3000
EPOCH_LEN = 25
TRAIN_EPISODES = 400
N_EVAL_TRACES = 30
# Typical per-epoch cost is O(100) (measured empirically); scale rewards down
# so DQN gradient steps are well-conditioned instead of exploding on raw
# cost magnitudes. Q-learning is scale-invariant so it is left unscaled.
DQN_REWARD_SCALE = 1.0 / 100.0

cell_grid = HexCellGrid(cell_radius=6.0)
cost_model = SignalingCostModel(lu_cost=10.0, paging_cost_per_cell=1.0)


def build_trace(key, cmr, seed):
    mobility = make_mobility(key, seed)
    return LUSimTrace(mobility, cell_grid, N_STEPS, cmr, seed=seed)


def make_switching_trace(seed):
    rng = np.random.default_rng(seed)
    segments = sample_switching_config(rng, MOBILITY_KEYS, CMR_VALUES, N_STEPS)
    return build_nonstationary_trace(segments, cell_grid, cost_model, seed), segments


# ---------- static baselines, evaluated two ways ----------

def realistic_fixed_threshold(segments, seed, mode):
    """Operator tunes the threshold using ONLY the first-segment regime
    (as would happen in practice, since they don't know the future shift),
    then that threshold is used for the whole trace."""
    key0, cmr0, steps0 = segments[0]
    first_seg_trace = build_trace(key0, cmr0, seed)  # separate trace for tuning only
    best_t, best_cost = None, None
    for t in THRESHOLD_RANGE:
        r = run_static_threshold(first_seg_trace, cost_model, t, mode=mode)
        if best_cost is None or r["total_cost"] < best_cost:
            best_cost, best_t = r["total_cost"], t
    return best_t


def realistic_fixed_time(segments, seed):
    key0, cmr0, steps0 = segments[0]
    first_seg_trace = build_trace(key0, cmr0, seed)
    best_p, best_cost = None, None
    for p in TIME_PERIOD_RANGE:
        res = run_time_based(first_seg_trace, cost_model, p, V_MAX_ASSUMED, ADJACENT_CELL_DIST)
        if best_cost is None or res["total_cost"] < best_cost:
            best_cost, best_p = res["total_cost"], p
    return best_p


def oracle_fixed_threshold(trace, mode):
    best_t, best_cost = None, None
    for t in THRESHOLD_RANGE:
        r = run_static_threshold(trace, cost_model, t, mode=mode)
        if best_cost is None or r["total_cost"] < best_cost:
            best_cost, best_t = r["total_cost"], t
    return best_cost


def oracle_fixed_time(trace):
    best_cost = None
    for p in TIME_PERIOD_RANGE:
        res = run_time_based(trace, cost_model, p, V_MAX_ASSUMED, ADJACENT_CELL_DIST)
        if best_cost is None or res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
    return best_cost


# ---------- RL training on switching episodes ----------

def train_agent_switching(agent_cls, agent_kwargs, seed_base=123, episodes=TRAIN_EPISODES):
    agent = agent_cls(**agent_kwargs)
    rng = np.random.default_rng(seed_base)
    for ep in range(episodes):
        seed = int(rng.integers(0, 10 ** 6))
        trace, _ = make_switching_trace(seed)
        env = AdaptiveLUEnv(trace, cost_model, epoch_len=EPOCH_LEN)
        obs = env.reset()
        done = False
        while not done:
            if isinstance(agent, QLearningAgent):
                action_idx, state = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.update(state, action_idx, reward, next_obs, done)
            else:
                action_idx = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.remember(obs, action_idx, reward * DQN_REWARD_SCALE, next_obs, done)
                agent.train_step()
            obs = next_obs
    return agent


def train_agent_stationary_only(agent_cls, agent_kwargs, seed_base=321, episodes=TRAIN_EPISODES):
    """Ablation: agent trained only ever seeing ONE regime per episode
    (as in the first experiment), then tested on switching traces it
    never saw during training."""
    agent = agent_cls(**agent_kwargs)
    rng = np.random.default_rng(seed_base)
    for ep in range(episodes):
        cmr = rng.choice(CMR_VALUES)
        key = rng.choice(MOBILITY_KEYS)
        seed = int(rng.integers(0, 10 ** 6))
        trace = build_trace(str(key), float(cmr), seed)
        env = AdaptiveLUEnv(trace, cost_model, epoch_len=EPOCH_LEN)
        obs = env.reset()
        done = False
        while not done:
            if isinstance(agent, QLearningAgent):
                action_idx, state = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.update(state, action_idx, reward, next_obs, done)
            else:
                action_idx = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.remember(obs, action_idx, reward * DQN_REWARD_SCALE, next_obs, done)
                agent.train_step()
            obs = next_obs
    return agent


def eval_agent_on_trace(agent, trace):
    env = AdaptiveLUEnv(trace, cost_model, epoch_len=EPOCH_LEN)
    obs = env.reset()
    done = False
    total_cost = 0.0
    while not done:
        if isinstance(agent, QLearningAgent):
            action_idx, _ = agent.act(obs, greedy=True)
        else:
            action_idx = agent.act(obs, greedy=True)
        obs, reward, done, info = env.step(ACTIONS[action_idx])
        total_cost += info["total_cost"]
    return total_cost


def main():
    print("Training RL agents on switching (non-stationary) episodes...")
    q_switch = train_agent_switching(QLearningAgent, {"seed": 42, "n_bins": 5}, episodes=300)
    dqn_switch = train_agent_switching(DQNAgent, {"seed": 42})

    print("Training RL agents on stationary-only episodes (ablation baseline)...")
    q_stat = train_agent_stationary_only(QLearningAgent, {"seed": 42, "n_bins": 5}, episodes=300)
    dqn_stat = train_agent_stationary_only(DQNAgent, {"seed": 42})

    rows = []
    # Held-out TEST seeds -- distinct from the seed=999/+500000 validation set
    # used throughout hyperparameter tuning above, so these final numbers are
    # not overfit to the traces used for model selection.
    eval_seed_rng = np.random.default_rng(24680)
    for i in range(N_EVAL_TRACES):
        seed = int(eval_seed_rng.integers(0, 10 ** 6)) + 900000  # disjoint from training and validation seeds
        trace, segments = make_switching_trace(seed)

        realistic_mv = realistic_fixed_threshold(segments, seed, "movement")
        realistic_ds = realistic_fixed_threshold(segments, seed, "distance")
        realistic_tm_p = realistic_fixed_time(segments, seed)

        realistic_mv_cost = run_static_threshold(trace, cost_model, realistic_mv, mode="movement")["total_cost"]
        realistic_ds_cost = run_static_threshold(trace, cost_model, realistic_ds, mode="distance")["total_cost"]
        realistic_tm_cost = run_time_based(trace, cost_model, realistic_tm_p, V_MAX_ASSUMED, ADJACENT_CELL_DIST)["total_cost"]

        oracle_mv_cost = oracle_fixed_threshold(trace, "movement")
        oracle_ds_cost = oracle_fixed_threshold(trace, "distance")
        oracle_tm_cost = oracle_fixed_time(trace)

        q_switch_cost = eval_agent_on_trace(q_switch, trace)
        dqn_switch_cost = eval_agent_on_trace(dqn_switch, trace)
        q_stat_cost = eval_agent_on_trace(q_stat, trace)
        dqn_stat_cost = eval_agent_on_trace(dqn_stat, trace)

        for scheme, cost in [
            ("Static Movement (tuned on pre-switch regime)", realistic_mv_cost),
            ("Static Distance (tuned on pre-switch regime)", realistic_ds_cost),
            ("Static Time (tuned on pre-switch regime)", realistic_tm_cost),
            ("Static Movement (oracle full-trace)", oracle_mv_cost),
            ("Static Distance (oracle full-trace)", oracle_ds_cost),
            ("Static Time (oracle full-trace)", oracle_tm_cost),
            ("AM-RL-LU Q-learning (trained on switching)", q_switch_cost),
            ("AM-RL-LU DQN (trained on switching)", dqn_switch_cost),
            ("AM-RL-LU Q-learning (trained stationary-only, ablation)", q_stat_cost),
            ("AM-RL-LU DQN (trained stationary-only, ablation)", dqn_stat_cost),
        ]:
            rows.append({"eval_trace": i, "seed": seed, "scheme": scheme, "total_cost": cost})

        if (i + 1) % 5 == 0:
            print(f"  evaluated {i+1}/{N_EVAL_TRACES} switching traces")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "nonstationary_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(df.groupby("scheme")["total_cost"].mean().sort_values())
    return df


if __name__ == "__main__":
    main()
