from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.config import (
    CORRELATION_MAX_RADIUS_MPC,
    CORRELATION_NBINS,
    CORRELATION_SUBSAMPLE,
    DBSCAN_EPS_MPC,
    DBSCAN_MIN_SAMPLES,
    FILAMENT_DENSITY_PERCENTILE,
    RANDOM_SEED,
    VOID_DENSITY_PERCENTILE,
)


def fit_cluster_labels(positions: np.ndarray) -> np.ndarray:
    return DBSCAN(
        eps=DBSCAN_EPS_MPC,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(positions)


def cluster_summary(labels: np.ndarray) -> dict:
    unique = [c for c in set(labels.tolist()) if c != -1]
    sizes = sorted([(labels == c).sum() for c in unique], reverse=True)
    n_clusters = len(unique)
    n_noise = int((labels == -1).sum())
    assigned = int(len(labels) - n_noise)
    return {
        "n_clusters": n_clusters,
        "n_noise_galaxies": n_noise,
        "fraction_clustered": float(assigned / max(len(labels), 1)),
        "cluster_sizes_top10": [int(s) for s in sizes[:10]],
        "largest_cluster_fraction": float(sizes[0] / max(len(labels), 1)) if sizes else 0.0,
    }


def void_filament_masks(density: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    occupied = density > 0
    vals = density[occupied]
    void_thr = np.percentile(vals, VOID_DENSITY_PERCENTILE)
    fil_thr = np.percentile(vals, FILAMENT_DENSITY_PERCENTILE)
    void_mask = occupied & (density <= void_thr)
    filament_mask = density >= fil_thr
    stats = {
        "void_voxel_count": int(void_mask.sum()),
        "filament_voxel_count": int(filament_mask.sum()),
        "void_threshold": float(void_thr),
        "filament_threshold": float(fil_thr),
        "void_fraction_of_occupied": float(void_mask.sum() / max(occupied.sum(), 1)),
        "filament_fraction_of_grid": float(filament_mask.sum() / density.size),
        "mean_occupancy_density": float(vals.mean()),
    }
    return void_mask, filament_mask, stats


def random_uniform_catalog(positions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    return rng.uniform(lo, hi, size=positions.shape).astype(np.float32)


def _pair_histogram(
    positions: np.ndarray,
    subsample: int,
    n_bins: int,
    r_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(positions)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        pts = positions[idx]
    else:
        pts = positions

    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(pts), k=1)
    dists = dists[iu]
    dists = dists[(dists > 0) & (dists < r_max)]

    bins = np.logspace(np.log10(0.5), np.log10(r_max), n_bins)
    counts, edges = np.histogram(dists, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    norm_counts = counts / (shell_volume + 1e-12)
    return centers, counts.astype(np.float64), norm_counts


def two_point_correlation(
    positions: np.ndarray,
    subsample: int = CORRELATION_SUBSAMPLE,
    n_bins: int = CORRELATION_NBINS,
    r_max: float = CORRELATION_MAX_RADIUS_MPC,
    seed: int = RANDOM_SEED,
) -> dict:
    centers, raw_counts, norm_counts = _pair_histogram(
        positions, subsample, n_bins, r_max, seed
    )
    xi_proxy = norm_counts / (norm_counts.mean() + 1e-12) - 1.0
    return {
        "r_mpc_centers": centers.tolist(),
        "pair_counts": raw_counts.tolist(),
        "xi_proxy": xi_proxy.tolist(),
    }


def correlation_vs_random(
    data_positions: np.ndarray,
    n_random_trials: int,
    seed: int = RANDOM_SEED,
) -> dict:
    data = two_point_correlation(data_positions, seed=seed)
    centers = np.array(data["r_mpc_centers"])
    xi_random_stack = []
    for t in range(n_random_trials):
        rnd = random_uniform_catalog(data_positions, seed=seed + 1000 + t)
        rnd_corr = two_point_correlation(rnd, seed=seed + 2000 + t)
        xi_random_stack.append(rnd_corr["xi_proxy"])
    xi_random = np.mean(xi_random_stack, axis=0)
    xi_data = np.array(data["xi_proxy"])
    xi_excess = xi_data - xi_random
    small = centers < 5.0
    strength = float(np.mean(xi_excess[small])) if small.any() else 0.0
    return {
        "r_mpc_centers": data["r_mpc_centers"],
        "xi_data": xi_data.tolist(),
        "xi_random": xi_random.tolist(),
        "xi_excess": xi_excess.tolist(),
        "clustering_strength_small_scale": strength,
    }


def nearest_neighbor_stats(positions: np.ndarray, k: int = 2) -> dict:
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    dists, _ = nbrs.kneighbors(positions)
    nn = dists[:, 1]
    return {
        "mean_nn_mpc": float(nn.mean()),
        "median_nn_mpc": float(np.median(nn)),
        "std_nn_mpc": float(nn.std()),
        "p10_nn_mpc": float(np.percentile(nn, 10)),
        "p90_nn_mpc": float(np.percentile(nn, 90)),
    }


def volume_geometry(positions: np.ndarray) -> dict:
    span = positions.max(axis=0) - positions.min(axis=0)
    volume = float(np.prod(span))
    density_gal_per_mpc3 = len(positions) / max(volume, 1e-6)
    return {
        "box_span_mpc": [float(s) for s in span],
        "box_volume_mpc3": volume,
        "number_density_per_mpc3": density_gal_per_mpc3,
    }
