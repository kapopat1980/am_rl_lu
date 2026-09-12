"""
Tabular Q-learning agent for the adaptive movement-based LU threshold.

State (from AdaptiveLUEnv._observe): [crossing_rate, call_rate, threshold_norm]
each discretized into bins. Action: {0: decrease, 1: hold, 2: increase}
mapped to {-1, 0, +1} threshold deltas.
"""
import numpy as np


ACTIONS = [-3, -1, 0, 1, 3]


class QLearningAgent:
    def __init__(self, n_bins=8, n_actions=5, alpha=0.15, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=4000, seed=None):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.rng = np.random.default_rng(seed)
        self.q_table = {}
        self.step_count = 0

    def _epsilon(self):
        frac = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _discretize(self, obs):
        cr_bin = min(self.n_bins - 1, int(obs[0] * self.n_bins))
        cl_bin = min(self.n_bins - 1, int(obs[1] * self.n_bins * 5))
        th_bin = min(self.n_bins - 1, int(obs[2] * self.n_bins))
        return (cr_bin, cl_bin, th_bin)

    def _q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def act(self, obs, greedy=False):
        state = self._discretize(obs)
        if not greedy and self.rng.random() < self._epsilon():
            action_idx = self.rng.integers(0, self.n_actions)
        else:
            action_idx = int(np.argmax(self._q(state)))
        self.step_count += 1
        return action_idx, state

    def update(self, state, action_idx, reward, next_obs, done):
        next_state = self._discretize(next_obs)
        q_sa = self._q(state)[action_idx]
        target = reward if done else reward + self.gamma * np.max(self._q(next_state))
        self._q(state)[action_idx] += self.alpha * (target - q_sa)
