"""
For each static LU scheme (movement-based, distance-based, time-based),
grid-search the best fixed threshold for a given trace, so that the RL
comparison is against each baseline's *best case*, not an arbitrary
fixed value (a common weakness in weaker LU papers).
"""
from experiments.environment import run_static_threshold, run_time_based


def best_static(trace, cost_model, mode, threshold_range):
    best = None
    for t in threshold_range:
        result = run_static_threshold(trace, cost_model, t, mode=mode)
        if best is None or result["total_cost"] < best["total_cost"]:
            best = result
    return best


def best_time_based(trace, cost_model, period_range, v_max, adjacent_cell_dist):
    best = None
    for p in period_range:
        result = run_time_based(trace, cost_model, p, v_max, adjacent_cell_dist)
        if best is None or result["total_cost"] < best["total_cost"]:
            best = result
    return best
