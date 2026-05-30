from __future__ import annotations

import numpy as np

from src.config import DENSITY_NPY, GRID_SIZE, POSITIONS_NPY


def positions_to_density_grid(
    positions: np.ndarray,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    eps = 1e-6
    scaled = (positions - lo) / (hi - lo + eps)
    scaled = np.clip(scaled, 0.0, 1.0 - 1e-8)
    indices = (scaled * (grid_size - 1)).astype(np.int32)

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)

    if grid.max() > 0:
        grid /= grid.max()
    return grid


def main() -> None:
    if not POSITIONS_NPY.exists():
        raise FileNotFoundError(
            f"Missing {POSITIONS_NPY}. Run: python -m src.coordinates"
        )

    positions = np.load(POSITIONS_NPY)
    grid = positions_to_density_grid(positions)
    np.save(DENSITY_NPY, grid)

    occupied = (grid > 0).sum()
    total = grid.size
    print(f"Saved density grid {grid.shape} -> {DENSITY_NPY}")
    print(f"  Occupied voxels: {occupied:,} / {total:,} ({100*occupied/total:.2f}%)")


if __name__ == "__main__":
    main()
