from __future__ import annotations

import json

import numpy as np

from src.analysis_core import (
    cluster_summary,
    correlation_vs_random,
    fit_cluster_labels,
    nearest_neighbor_stats,
    void_filament_masks,
    volume_geometry,
)
from src.config import (
    CLUSTER_LABELS_NPY,
    CORRELATION_RANDOM_TRIALS,
    DATA_DIR,
    DENSITY_NPY,
    FILAMENT_MASK_NPY,
    POSITIONS_NPY,
    RESULTS_DIR,
    VOID_MASK_NPY,
)


def main() -> None:
    if not POSITIONS_NPY.exists() or not DENSITY_NPY.exists():
        raise FileNotFoundError(
            "Run coordinates + density first: "
            "python -m src.coordinates then python -m src.density"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    positions = np.load(POSITIONS_NPY)
    density = np.load(DENSITY_NPY)

    labels = fit_cluster_labels(positions)
    void_mask, filament_mask, vf_stats = void_filament_masks(density)

    np.save(CLUSTER_LABELS_NPY, labels)
    np.save(VOID_MASK_NPY, void_mask)
    np.save(FILAMENT_MASK_NPY, filament_mask)

    report = {
        "n_galaxies": int(len(positions)),
        "geometry": volume_geometry(positions),
        "nearest_neighbors": nearest_neighbor_stats(positions),
        "clusters": cluster_summary(labels),
        "density_structure": vf_stats,
        "correlation": correlation_vs_random(
            positions, n_random_trials=CORRELATION_RANDOM_TRIALS
        ),
    }

    out = RESULTS_DIR / "patterns.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pattern report -> {out}")
    print(f"Cluster labels -> {CLUSTER_LABELS_NPY}")
    cs = report["correlation"]["clustering_strength_small_scale"]
    print(f"Small-scale clustering strength (xi_data - xi_random): {cs:.3f}")


if __name__ == "__main__":
    main()
