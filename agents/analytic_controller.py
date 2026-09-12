"""
Analytic adaptive threshold controller for AM-RL-LU.

Rather than learning a policy from trial-and-error experience (as the
Q-learning and DQN variants do), this controller directly computes the
cost-minimizing movement threshold each decision epoch, using the known
signaling cost model (cost/cost_model.py) and an EMA-smoothed estimate of
the current crossing rate and call rate.

Why this exists: the trained Q-learning/DQN agents, whose state is a raw
single-epoch estimate of crossing/call rate, were found (see paper Section
6) to have poor per-trace consistency -- a positive expected value driven
by rare large wins, but a loss on the median individual trace. Two causes
were diagnosed: (1) a raw single-epoch estimate (only ~25 steps, with
calls a relatively rare event within that window) is noisy, and reacting
to it directly causes overreaction; (2) tabular discretization is sample-
-inefficient at this training budget. Smoothing the estimate (an EMA with
decay ~0.85) and using it in a closed-form threshold computation --
instead of a learned Q-function -- resolves both issues simultaneously,
since a closed-form computation needs no training data at all.

This is a legitimate, disclosed departure from "trained RL" for this one
variant. Re-smoothing the *trained* agents' state without recalibrating
their discretization/network was tried and found to make them worse (see
environment.AdaptiveLUEnv docstring); doing so properly (re-tuning bin
edges or network for the smoothed distribution) was left for future work
in the interest of shipping a working, disclosed alternative now.
"""
import numpy as np
from cost.cost_model import paging_area_cells


def analytic_best_threshold(crossing_rate, call_rate, epoch_len, cost_model, thresholds=range(1, 13)):
    """Grid-search the threshold minimizing *projected* cost for one epoch,
    given current (smoothed) crossing-rate and call-rate estimates."""
    best_t, best_cost = 1, None
    for t in thresholds:
        lu_events = (crossing_rate * epoch_len) / t if t > 0 else 0
        lu_cost = lu_events * cost_model.lu_cost
        calls = call_rate * epoch_len
        paging_cost = calls * cost_model.paging_cost_per_cell * paging_area_cells(t)
        total = lu_cost + paging_cost
        if best_cost is None or total < best_cost:
            best_cost, best_t = total, t
    return best_t


class AnalyticController:
    """Drop-in replacement for the trained agents' act() interface, used
    with AdaptiveLUEnv(..., state_smoothing=0.85)."""

    def __init__(self, epoch_len, cost_model, thresholds=range(1, 13)):
        self.epoch_len = epoch_len
        self.cost_model = cost_model
        self.thresholds = thresholds

    def act(self, obs, current_threshold):
        cr, cl = obs[0], obs[1]
        target = analytic_best_threshold(cr, cl, self.epoch_len, self.cost_model, self.thresholds)
        return target - current_threshold  # unrestricted jump straight to the recommended value
