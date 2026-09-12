"""
Mobility models for the AM-RL-LU simulator.

Each model produces a sequence of (x, y) positions over discrete time steps
for a single mobile user, on a plane covered by a hexagonal cell grid.
All models are self-contained, deterministic given a seed, and validated
against known statistical properties (see tests/test_mobility.py).
"""
import numpy as np


class RandomWalkMobility:
    """Discrete random walk: at each step, pick a random direction and
    move a fixed step length. Represents low-predictability, pedestrian-like
    movement."""

    def __init__(self, step_length=1.0, seed=None):
        self.step_length = step_length
        self.rng = np.random.default_rng(seed)

    def generate(self, n_steps, start=(0.0, 0.0)):
        positions = np.zeros((n_steps + 1, 2))
        positions[0] = start
        angles = self.rng.uniform(0, 2 * np.pi, size=n_steps)
        dx = self.step_length * np.cos(angles)
        dy = self.step_length * np.sin(angles)
        positions[1:, 0] = start[0] + np.cumsum(dx)
        positions[1:, 1] = start[1] + np.cumsum(dy)
        return positions


class FluidFlowMobility:
    """Fluid flow model: constant speed and direction, with direction
    resampled after an exponentially distributed dwell time. Represents
    high-mobility, highway-like movement with long straight segments."""

    def __init__(self, speed=3.0, mean_dwell=20.0, seed=None):
        self.speed = speed
        self.mean_dwell = mean_dwell
        self.rng = np.random.default_rng(seed)

    def generate(self, n_steps, start=(0.0, 0.0)):
        positions = np.zeros((n_steps + 1, 2))
        positions[0] = start
        pos = np.array(start, dtype=float)
        angle = self.rng.uniform(0, 2 * np.pi)
        steps_left = max(1, int(self.rng.exponential(self.mean_dwell)))
        for t in range(1, n_steps + 1):
            if steps_left <= 0:
                angle = self.rng.uniform(0, 2 * np.pi)
                steps_left = max(1, int(self.rng.exponential(self.mean_dwell)))
            pos = pos + self.speed * np.array([np.cos(angle), np.sin(angle)])
            positions[t] = pos
            steps_left -= 1
        return positions


class GaussMarkovMobility:
    """Gauss-Markov model: velocity evolves as a correlated (Markov) process,
    controlled by a memory parameter alpha in [0, 1]. alpha=0 reduces to a
    memoryless random walk in velocity; alpha=1 gives constant-velocity
    linear motion. Represents an intermediate, realistic mobility regime
    (e.g. urban pedestrian-to-vehicular mix)."""

    def __init__(self, alpha=0.75, mean_speed=2.0, speed_std=0.5,
                 mean_angle=0.0, angle_std=0.4, seed=None):
        self.alpha = alpha
        self.mean_speed = mean_speed
        self.speed_std = speed_std
        self.mean_angle = mean_angle
        self.angle_std = angle_std
        self.rng = np.random.default_rng(seed)

    def generate(self, n_steps, start=(0.0, 0.0)):
        positions = np.zeros((n_steps + 1, 2))
        positions[0] = start
        pos = np.array(start, dtype=float)
        speed = self.mean_speed
        angle = self.mean_angle
        for t in range(1, n_steps + 1):
            speed = (self.alpha * speed
                     + (1 - self.alpha) * self.mean_speed
                     + np.sqrt(1 - self.alpha ** 2) * self.rng.normal(0, self.speed_std))
            angle = (self.alpha * angle
                     + (1 - self.alpha) * self.mean_angle
                     + np.sqrt(1 - self.alpha ** 2) * self.rng.normal(0, self.angle_std))
            speed = max(speed, 0.0)
            pos = pos + speed * np.array([np.cos(angle), np.sin(angle)])
            positions[t] = pos
        return positions


MOBILITY_MODELS = {
    "random_walk": RandomWalkMobility,
    "fluid_flow": FluidFlowMobility,
    "gauss_markov": GaussMarkovMobility,
}
