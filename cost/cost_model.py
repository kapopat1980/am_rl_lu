"""
Signaling cost model shared by all LU strategies (static baselines and
the RL-adaptive scheme), so that comparisons are apples-to-apples.

Cost components (standard formulation used in LU/paging literature):
  - Location Update (LU) cost:  C_u, incurred each time the user performs
    a location update (crosses the movement/distance/time threshold).
  - Paging cost: C_p per cell searched when a call arrives; total paging
    cost for a call = C_p * (number of cells in the paging area).

Total cost over an observation window = LU_count * C_u + Call_count * C_p * paging_area_size

Paging area size is determined by the *ring radius* used for paging
(radius = movement threshold for movement-based LU): a paging area of
ring radius k covers 1 + 3k(k+1) hex cells.
"""
import numpy as np


def paging_area_cells(ring_radius):
    """Number of hex cells in a disk of the given ring radius (radius 0 = 1 cell)."""
    k = max(0, int(ring_radius))
    return 1 + 3 * k * (k + 1)


class SignalingCostModel:
    def __init__(self, lu_cost=10.0, paging_cost_per_cell=1.0, max_paging_delay_cells=None,
                 delay_penalty=0.0):
        """
        lu_cost: cost of a single location update transaction.
        paging_cost_per_cell: cost of paging a single cell.
        max_paging_delay_cells: optional cap; if the paging area implied by the
            threshold exceeds this many cells, an extra delay_penalty is added
            (models a call-delivery-delay constraint).
        delay_penalty: additional cost applied per call if the paging area is
            too large (SLA-style constraint), 0 disables this term.
        """
        self.lu_cost = lu_cost
        self.paging_cost_per_cell = paging_cost_per_cell
        self.max_paging_delay_cells = max_paging_delay_cells
        self.delay_penalty = delay_penalty

    def call_cost(self, ring_radius):
        cells = paging_area_cells(ring_radius)
        cost = self.paging_cost_per_cell * cells
        if self.max_paging_delay_cells is not None and cells > self.max_paging_delay_cells:
            cost += self.delay_penalty
        return cost, cells

    def total_cost(self, n_location_updates, n_calls, ring_radius):
        lu_total = n_location_updates * self.lu_cost
        call_unit_cost, cells = self.call_cost(ring_radius)
        paging_total = n_calls * call_unit_cost
        return lu_total + paging_total, lu_total, paging_total, cells
