"""
A minimal, dependency-free DQN implementation (numpy only), used as the
"deeper" ML-adaptive variant of AM-RL-LU, compared against the tabular
Q-learning baseline.

Network: single hidden layer MLP, manual forward pass and manual
backprop, updated with Adam (not plain SGD -- plain SGD was found to
converge to a degenerate "always decrease threshold" policy: the
network learned an accurate state-value estimate but too flat/inaccurate
an action-advantage estimate to discriminate between actions well.
Adam's per-parameter adaptive step sizes fix this in practice).
Includes an experience replay buffer and a periodically-synced target
network, matching the standard DQN recipe (Mnih et al. 2015) as far as
is practical in a numpy-only setting.
"""
import numpy as np
from collections import deque


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class TinyMLP:
    """One hidden layer MLP: state_dim -> hidden -> n_actions, trained with Adam."""

    def __init__(self, state_dim, hidden_dim, n_actions, seed=None):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, size=(state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, size=(hidden_dim, n_actions))
        self.b2 = np.zeros(n_actions)
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        q = a1 @ self.W2 + self.b2
        cache = (x, z1, a1)
        return q, cache

    def backward(self, cache, dq, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        x, z1, a1 = cache
        dW2 = a1.T @ dq / x.shape[0]
        db2 = dq.mean(axis=0)
        da1 = dq @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x.T @ dz1 / x.shape[0]
        db1 = dz1.mean(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g ** 2)
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, k, getattr(self, k) - update)

    def copy_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()


class DQNAgent:
    def __init__(self, state_dim=3, hidden_dim=32, n_actions=5, gamma=0.95,
                 lr=0.003, buffer_size=10000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.15, epsilon_decay_steps=10000,
                 target_sync_every=200, seed=None):
        self.online = TinyMLP(state_dim, hidden_dim, n_actions, seed=seed)
        self.target = TinyMLP(state_dim, hidden_dim, n_actions, seed=seed)
        self.target.copy_from(self.online)
        self.gamma = gamma
        self.lr = lr
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_every = target_sync_every
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def _epsilon(self):
        frac = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs, greedy=False):
        self.step_count += 1
        if not greedy and self.rng.random() < self._epsilon():
            return int(self.rng.integers(0, self.n_actions))
        q, _ = self.online.forward(obs[None, :])
        return int(np.argmax(q[0]))

    def remember(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        idxs = self.rng.integers(0, len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idxs]
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        q_next_online, _ = self.online.forward(next_states)
        best_next_actions = np.argmax(q_next_online, axis=1)
        q_next_target, _ = self.target.forward(next_states)
        q_next_selected = q_next_target[np.arange(len(batch)), best_next_actions]
        target_q = rewards + self.gamma * (1 - dones) * q_next_selected

        q_pred, cache = self.online.forward(states)
        dq = np.zeros_like(q_pred)
        td_error = q_pred[np.arange(len(batch)), actions] - target_q
        dq[np.arange(len(batch)), actions] = td_error
        self.online.backward(cache, dq, self.lr)

        if self.step_count % self.target_sync_every == 0:
            self.target.copy_from(self.online)
