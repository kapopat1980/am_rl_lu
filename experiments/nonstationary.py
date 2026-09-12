"""
Non-stationary trace generation: builds a single trace made of two segments
with (possibly) different mobility models and/or CMR values, joined
continuously in position. This is the scenario that actually tests whether
an RL-adaptive threshold has value over a static one: a static scheme must
commit to one threshold for the whole trace, while an online agent can
detect the shift (via its epoch-level movement/call-rate estimates) and
re-tune.
"""
import numpy as np


def build_nonstationary_trace(segments, cell_grid, cost_model_unused, seed):
    """
    segments: list of (mobility_key, cmr, n_steps) tuples, played in sequence.
    Returns a lightweight object exposing .cells, .calls, .n_steps, .segments,
    .switch_points -- the same interface LUSimTrace exposes, so it plugs
    directly into run_static_threshold / run_time_based / AdaptiveLUEnv.
    """
    from experiments.run_experiment import make_mobility  # local import avoids cycle

    rng_master = np.random.default_rng(seed)
    positions_chunks = []
    start = (0.0, 0.0)
    switch_points = [0]
    for key, cmr, steps in segments:
        seed_i = int(rng_master.integers(0, 10 ** 6))
        mobility = make_mobility(key, seed_i)
        pos = mobility.generate(steps, start=start)
        positions_chunks.append(pos if not positions_chunks else pos[1:])
        start = tuple(pos[-1])
        switch_points.append(switch_points[-1] + steps)

    positions = np.concatenate(positions_chunks, axis=0)
    cells = cell_grid.cell_sequence(positions)
    n_steps = len(cells) - 1

    calls = np.zeros(n_steps + 1, dtype=bool)
    rng_calls = np.random.default_rng(seed + 999983)
    cum = 0
    for key, cmr, steps in segments:
        seg_crossings = sum(
            1 for t in range(cum + 1, cum + steps + 1) if cells[t] != cells[t - 1]
        )
        crossing_rate = seg_crossings / steps if steps > 0 else 0.0
        call_prob = min(1.0, cmr * crossing_rate) if crossing_rate > 0 else cmr / max(1, steps)
        seg_calls = rng_calls.random(steps) < call_prob
        calls[cum + 1: cum + steps + 1] = seg_calls
        cum += steps

    class NonStationaryTrace:
        pass

    trace = NonStationaryTrace()
    trace.cells = cells
    trace.calls = calls
    trace.n_steps = n_steps
    trace.segments = segments
    trace.switch_points = switch_points
    return trace


def sample_switching_config(rng, mobility_keys, cmr_values, n_steps, min_frac=0.3, max_frac=0.7):
    switch_frac = rng.uniform(min_frac, max_frac)
    steps_a = int(n_steps * switch_frac)
    steps_b = n_steps - steps_a
    key_a = rng.choice(mobility_keys)
    key_b = rng.choice(mobility_keys)
    cmr_a = rng.choice(cmr_values)
    cmr_b = rng.choice(cmr_values)
    return [(str(key_a), float(cmr_a), steps_a), (str(key_b), float(cmr_b), steps_b)]
