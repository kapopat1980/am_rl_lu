"""
Hexagonal cell grid using axial coordinates.
Converts continuous (x, y) positions into discrete cell IDs, and provides
ring-distance (used for paging area sizing) between cells.
"""
import numpy as np


class HexCellGrid:
    def __init__(self, cell_radius=1.0):
        """cell_radius: the 'radius' of each hexagonal cell (center to vertex)."""
        self.cell_radius = cell_radius
        # width/height of a pointy-top hex layout
        self._w = np.sqrt(3) * cell_radius
        self._h = 2 * cell_radius

    def xy_to_axial(self, x, y):
        """Convert continuous coordinates to axial hex coordinates (q, r)."""
        q = (np.sqrt(3) / 3 * x - 1 / 3 * y) / self.cell_radius
        r = (2 / 3 * y) / self.cell_radius
        return self._axial_round(q, r)

    @staticmethod
    def _axial_round(q, r):
        x = q
        z = r
        y = -x - z
        rx, ry, rz = round(x), round(y), round(z)
        x_diff, y_diff, z_diff = abs(rx - x), abs(ry - y), abs(rz - z)
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        return int(rx), int(rz)

    def cell_id(self, x, y):
        q, r = self.xy_to_axial(x, y)
        return (q, r)

    def cell_sequence(self, positions):
        """Given an (N, 2) array of positions, return the sequence of cell IDs."""
        return [self.cell_id(x, y) for x, y in positions]

    @staticmethod
    def ring_distance(cell_a, cell_b):
        """Hex distance (number of rings) between two axial cells."""
        qa, ra = cell_a
        qb, rb = cell_b
        return (abs(qa - qb) + abs(qa + ra - qb - rb) + abs(ra - rb)) // 1 // 2 \
            if False else HexCellGrid._axial_distance(cell_a, cell_b)

    @staticmethod
    def _axial_distance(a, b):
        aq, ar = a
        bq, br = b
        return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) / 2

    @staticmethod
    def cells_within_ring(center, ring):
        """Return list of axial cells within `ring` hops of center (a disk,
        used as the paging area of a given radius)."""
        cq, cr = center
        results = []
        for dq in range(-ring, ring + 1):
            for dr in range(max(-ring, -dq - ring), min(ring, -dq + ring) + 1):
                results.append((cq + dq, cr + dr))
        return results
