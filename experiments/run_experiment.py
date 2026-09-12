import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from mobility.models import MOBILITY_MODELS
from mobility.cellgrid import HexCellGrid
from cost.cost_model import SignalingCostModel
from experiments.environment import LUSimTrace, AdaptiveLUEnv
from experiments.baselines import best_static, best_time_based
from agents.q_learning import QLearningAgent, ACTIONS
from agents.dqn_numpy import DQNAgent

N_STEPS = 3000
EPOCH_LEN = 25
CMR_VALUES = [0.2, 0.5, 1.0, 2.0, 5.0]
MOBILITY_KEYS = ["random_walk", "fluid_flow", "gauss_markov"]
TRAIN_EPISODES = 60
EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]  # held-out evaluation seeds
THRESHOLD_RANGE = range(1, 13)
TIME_PERIOD_RANGE = range(5, 121, 5)

cell_grid = HexCellGrid(cell_radius=6.0)  # calibrated so a mobile crosses cells at a realistic
                                           # rate (~0.14-0.18 crossings/step) relative to its speed
cost_model = SignalingCostModel(lu_cost=10.0, paging_cost_per_cell=1.0)
# Conservative worst-case speed assumption used to derive the time-based
# scheme's paging radius from its timer period (see environment.time_based_paging_radius).
# Set above the fastest mobility model's typical speed (fluid_flow=1.5) with margin.
V_MAX_ASSUMED = 2.0
ADJACENT_CELL_DIST = np.sqrt(3) * cell_grid.cell_radius


def make_mobility(key, seed):
    if key == "random_walk":
        return MOBILITY_MODELS[key](step_length=1.2, seed=seed)
    if key == "fluid_flow":
        return MOBILITY_MODELS[key](speed=1.5, mean_dwell=15.0, seed=seed)
    if key == "gauss_markov":
        return MOBILITY_MODELS[key](alpha=0.75, mean_speed=1.3, speed_std=0.4, seed=seed)
    raise ValueError(key)


def build_trace(key, cmr, seed):
    mobility = make_mobility(key, seed)
    return LUSimTrace(mobility, cell_grid, N_STEPS, cmr, seed=seed)


def train_agent(agent_cls, mobility_key, agent_kwargs=None):
    agent_kwargs = agent_kwargs or {}
    agent = agent_cls(**agent_kwargs)
    rng = np.random.default_rng(7)
    for ep in range(TRAIN_EPISODES):
        cmr = rng.choice(CMR_VALUES)
        seed = int(rng.integers(0, 100000))
        trace = build_trace(mobility_key, cmr, seed)
        env = AdaptiveLUEnv(trace, cost_model, epoch_len=EPOCH_LEN)
        obs = env.reset()
        done = False
        while not done:
            if isinstance(agent, QLearningAgent):
                action_idx, state = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.update(state, action_idx, reward, next_obs, done)
            else:  # DQN
                action_idx = agent.act(obs)
                next_obs, reward, done, info = env.step(ACTIONS[action_idx])
                agent.remember(obs, action_idx, reward, next_obs, done)
                agent.train_step()
            obs = next_obs
    return agent


def eval_agent(agent, mobility_key, cmr, seed):
    trace = build_trace(mobility_key, cmr, seed)
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


def eval_static(mobility_key, cmr, seed, mode):
    trace = build_trace(mobility_key, cmr, seed)
    if mode == "time":
        result = best_time_based(trace, cost_model, TIME_PERIOD_RANGE, V_MAX_ASSUMED, ADJACENT_CELL_DIST)
    else:
        result = best_static(trace, cost_model, mode, THRESHOLD_RANGE)
    return result["total_cost"]


def main():
    rows = []
    for mobility_key in MOBILITY_KEYS:
        print(f"=== Training agents for mobility model: {mobility_key} ===")
        q_agent = train_agent(QLearningAgent, mobility_key, {"seed": 42})
        dqn_agent = train_agent(DQNAgent, mobility_key, {"seed": 42})

        for cmr in CMR_VALUES:
            for seed in EVAL_SEEDS:
                movement_cost = eval_static(mobility_key, cmr, seed, "movement")
                distance_cost = eval_static(mobility_key, cmr, seed, "distance")
                time_cost = eval_static(mobility_key, cmr, seed, "time")
                q_cost = eval_agent(q_agent, mobility_key, cmr, seed)
                dqn_cost = eval_agent(dqn_agent, mobility_key, cmr, seed)

                for scheme, cost in [
                    ("Static Movement-based", movement_cost),
                    ("Static Distance-based", distance_cost),
                    ("Static Time-based", time_cost),
                    ("AM-RL-LU (Q-learning)", q_cost),
                    ("AM-RL-LU (DQN)", dqn_cost),
                ]:
                    rows.append({
                        "mobility_model": mobility_key,
                        "cmr": cmr,
                        "seed": seed,
                        "scheme": scheme,
                        "total_cost": cost,
                    })
            print(f"  cmr={cmr} done")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "raw_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved raw results to {out_path}")
    return df


if __name__ == "__main__":
    main()
