"""
Core simulation engine for location-update (LU) strategies.

All strategies (static movement/distance/time-based baselines, and the
RL-adaptive AM-RL-LU scheme) run through this same engine so that cost
comparisons are apples-to-apples: identical mobility trace, identical
call arrival trace, identical cost model.

Design (documented explicitly for the paper's Methodology section):
  - "Paging radius" R is the guaranteed bound (in hex-cell rings) within
    which the network knows the mobile is located, since its last
    location update. Paging cost = C_p * cells_in_disk(R).
  - Movement-based: LU triggers when cumulative cell-crossings since the
    last update reach threshold T_m. R = T_m.
  - Distance-based: LU triggers when hex-distance from the last
    registered cell reaches threshold T_d. R = T_d.
  - Time-based: LU triggers every T_t simulation steps. Its paging radius
    R_t is treated as an independently-tunable operator parameter (grid
    searched, like T_m/T_d), since time-based LU does not itself bound
    displacement. This is a standard practical simplification, noted in
    the Methodology/Limitations section.
  - AM-RL-LU (ours): movement-based trigger mechanism, but T_m is
    re-set every decision epoch by an RL agent instead of held fixed.
"""
import numpy as np
from mobility.cellgrid import HexCellGrid


class LUSimTrace:
    """Precomputed mobility + call-arrival trace, reusable across strategies
    for a fair comparison (identical randomness per trial)."""

    def __init__(self, mobility_model, cell_grid: HexCellGrid, n_steps, cmr, seed=None):
        rng = np.random.default_rng(seed)
        positions = mobility_model.generate(n_steps)
        self.cells = cell_grid.cell_sequence(positions)
        self.grid = cell_grid
        # Call arrivals: CMR = calls per movement (cell crossing). We derive a
        # per-step call probability from the crossing rate observed in this
        # trace, so CMR is comparable across mobility models with different
        # absolute speeds.
        crossings = sum(1 for i in range(1, len(self.cells)) if self.cells[i] != self.cells[i - 1])
        crossing_rate = crossings / n_steps if n_steps > 0 else 0.0
        call_prob = min(1.0, cmr * crossing_rate) if crossing_rate > 0 else cmr / n_steps
        self.calls = rng.random(n_steps + 1) < call_prob
        self.n_steps = n_steps
        self.cmr = cmr
        self.crossing_rate = crossing_rate


def run_static_threshold(trace: LUSimTrace, cost_model, threshold, mode="movement"):
    """Run a static-threshold LU scheme (movement-based or distance-based)
    over a precomputed trace. Returns total cost breakdown."""
    anchor_cell = trace.cells[0]
    cross_count = 0
    n_lu = 0
    n_calls = 0
    call_cost_accum = 0.0
    for t in range(1, trace.n_steps + 1):
        moved = trace.cells[t] != trace.cells[t - 1]
        if moved:
            cross_count += 1
        if trace.calls[t]:
            n_calls += 1
            unit_cost, _ = cost_model.call_cost(threshold)
            call_cost_accum += unit_cost
        if mode == "movement":
            trigger = cross_count >= threshold
        elif mode == "distance":
            trigger = HexCellGrid._axial_distance(anchor_cell, trace.cells[t]) >= threshold
        else:
            raise ValueError(mode)
        if trigger:
            n_lu += 1
            cross_count = 0
            anchor_cell = trace.cells[t]
    lu_cost_total = n_lu * cost_model.lu_cost
    total = lu_cost_total + call_cost_accum
    return {"total_cost": total, "lu_cost": lu_cost_total, "paging_cost": call_cost_accum,
            "n_lu": n_lu, "n_calls": n_calls, "threshold": threshold}


def time_based_paging_radius(period, v_max, adjacent_cell_dist):
    """The paging radius for a time-based scheme is NOT a free parameter --
    it is the worst-case number of cell-rings a mobile travelling at up to
    v_max could have covered during one timer period, since the network has
    no other information bounding the mobile's displacement between updates.
    Using a smaller radius than this would mean the network incorrectly
    assumes it can find the user with a smaller search area than physically
    possible, which is not a valid deployment. Period remains the only
    free/tunable design parameter for this scheme."""
    return max(1, int(np.ceil(v_max * period / adjacent_cell_dist)))


def run_time_based(trace: LUSimTrace, cost_model, period, v_max, adjacent_cell_dist):
    paging_radius = time_based_paging_radius(period, v_max, adjacent_cell_dist)
    n_lu = 0
    n_calls = 0
    call_cost_accum = 0.0
    since_update = 0
    for t in range(1, trace.n_steps + 1):
        since_update += 1
        if trace.calls[t]:
            n_calls += 1
            unit_cost, _ = cost_model.call_cost(paging_radius)
            call_cost_accum += unit_cost
        if since_update >= period:
            n_lu += 1
            since_update = 0
    lu_cost_total = n_lu * cost_model.lu_cost
    total = lu_cost_total + call_cost_accum
    return {"total_cost": total, "lu_cost": lu_cost_total, "paging_cost": call_cost_accum,
            "n_lu": n_lu, "n_calls": n_calls, "threshold": period, "paging_radius": paging_radius}


class AdaptiveLUEnv:
    """RL-facing environment: movement-based LU trigger with a threshold that
    an external agent resets at every decision epoch. Mirrors run_static_threshold
    but exposes a step-by-epoch interface with observation/reward for RL agents.
    """

    def __init__(self, trace: LUSimTrace, cost_model, epoch_len=25, min_threshold=1, max_threshold=12,
                 state_smoothing=0.0):
        """state_smoothing: EMA decay applied to the crossing/call-rate estimates fed to the
        agent (see _observe). Default 0.0 preserves the original raw single-epoch estimate
        (used by the Q-learning/DQN agents, whose discretization/network were calibrated
        against that distribution). A smoothed estimate (state_smoothing=0.85) was found
        essential for the analytic controller (Section IV-C) but was NOT beneficial for the
        tabular/DQN agents as-is, since their existing calibration assumes the raw
        distribution -- re-smoothing without re-calibrating them made both noticeably worse."""
        self.trace = trace
        self.cost_model = cost_model
        self.epoch_len = epoch_len
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.state_smoothing = state_smoothing
        self.reset()

    def reset(self):
        self.t = 0
        self.anchor_cell = self.trace.cells[0]
        self.cross_count = 0
        self.threshold = (self.min_threshold + self.max_threshold) // 2
        self._recent_crossings = []
        self._recent_calls = []
        self._smoothed_cr = None
        self._smoothed_cl = None
        return self._observe()

    def _observe(self):
        # Estimate recent movement rate and call rate from the last epoch,
        # normalized so the agent generalizes across mobility regimes, and
        # EMA-smoothed across epochs to reduce single-epoch sampling noise.
        # The EMA is seeded with the first real observation (rather than 0)
        # to avoid a cold-start transient where an initial "no data yet"
        # estimate of 0 wrongly implies "no calls," recommending a wastefully
        # large threshold for the first epoch.
        cr = np.mean(self._recent_crossings) if self._recent_crossings else 0.0
        cl = np.mean(self._recent_calls) if self._recent_calls else 0.0
        s = self.state_smoothing
        if self._smoothed_cr is None:
            self._smoothed_cr, self._smoothed_cl = cr, cl
        else:
            self._smoothed_cr = s * self._smoothed_cr + (1 - s) * cr
            self._smoothed_cl = s * self._smoothed_cl + (1 - s) * cl
        thr_norm = (self.threshold - self.min_threshold) / max(1, (self.max_threshold - self.min_threshold))
        return np.array([self._smoothed_cr, self._smoothed_cl, thr_norm], dtype=np.float32)

    def done(self):
        return self.t >= self.trace.n_steps

    def step(self, action):
        """action in {-1, 0, +1}: decrease / hold / increase threshold."""
        self.threshold = int(np.clip(self.threshold + action, self.min_threshold, self.max_threshold))
        epoch_lu_cost = 0.0
        epoch_paging_cost = 0.0
        crossings_this_epoch = 0
        calls_this_epoch = 0
        steps_run = 0
        for _ in range(self.epoch_len):
            if self.done():
                break
            self.t += 1
            steps_run += 1
            moved = self.trace.cells[self.t] != self.trace.cells[self.t - 1]
            if moved:
                self.cross_count += 1
                crossings_this_epoch += 1
            if self.trace.calls[self.t]:
                calls_this_epoch += 1
                unit_cost, _ = self.cost_model.call_cost(self.threshold)
                epoch_paging_cost += unit_cost
            if self.cross_count >= self.threshold:
                epoch_lu_cost += self.cost_model.lu_cost
                self.cross_count = 0
                self.anchor_cell = self.trace.cells[self.t]
        self._recent_crossings = [crossings_this_epoch / max(1, steps_run)]
        self._recent_calls = [calls_this_epoch / max(1, steps_run)]
        total_epoch_cost = epoch_lu_cost + epoch_paging_cost
        reward = -total_epoch_cost
        info = {"lu_cost": epoch_lu_cost, "paging_cost": epoch_paging_cost,
                "total_cost": total_epoch_cost}
        return self._observe(), reward, self.done(), info
