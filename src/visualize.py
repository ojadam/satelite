from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.config import (
    CLUSTER_LABELS_NPY,
    DENSITY_NPY,
    FIGURE_DPI,
    FILAMENT_MASK_NPY,
    PLOT_SUBSAMPLE,
    POSITIONS_NPY,
    RESULTS_DIR,
    VOID_MASK_NPY,
)

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#6b7280",
    "axes.labelcolor": "#e5e7eb",
    "text.color": "#e5e7eb",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
    "grid.color": "#374151",
    "font.size": 10,
})


def subsample_indices(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap, dtype=int)


def save_correlation_comparison(report: dict, path) -> None:
    cr = report["correlation"]
    r = cr["r_mpc_centers"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, cr["xi_data"], "o-", ms=3, lw=1.5, label="SDSS galaxies", color="#60a5fa")
    ax.plot(r, cr["xi_random"], "s-", ms=2, lw=1, label="Random uniform (same box)", color="#f87171", alpha=0.85)
    ax.plot(r, cr["xi_excess"], "^-", ms=2, lw=1.2, label="Excess (data − random)", color="#34d399")
    ax.axhline(0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Separation r (Mpc)")
    ax.set_ylabel("ξ proxy")
    ax.set_title("Two-point correlation: clustered vs random placement")
    ax.legend(framealpha=0.2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_cluster_plot(positions: np.ndarray, labels: np.ndarray, path) -> None:
    idx = subsample_indices(len(positions), PLOT_SUBSAMPLE)
    pts = positions[idx]
    lab = labels[idx]
    noise = lab == -1
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(
        pts[noise, 0], pts[noise, 1], s=0.5, c="#4b5563", alpha=0.25, edgecolors="none", label="Noise",
    )
    clustered = ~noise
    sc = ax.scatter(
        pts[clustered, 0],
        pts[clustered, 1],
        s=1.2,
        c=lab[clustered],
        cmap="turbo",
        alpha=0.65,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Cluster ID", shrink=0.8)
    ax.set_xlabel("x (comoving Mpc)")
    ax.set_ylabel("y (comoving Mpc)")
    ax.set_title("Galaxy positions colored by DBSCAN cluster")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_density_structure(density: np.ndarray, void_mask: np.ndarray, filament_mask: np.ndarray, path) -> None:
    mid = density.shape[0] // 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    im0 = axes[0, 0].imshow(density[mid], origin="lower", cmap="magma")
    axes[0, 0].set_title(f"Density slice (z={mid})")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(void_mask[mid], origin="lower", cmap=ListedColormap(["#0f1117", "#1d4ed8"]))
    axes[0, 1].set_title("Void proxy (low-density voxels)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(filament_mask[mid], origin="lower", cmap=ListedColormap(["#0f1117", "#f59e0b"]))
    axes[1, 0].set_title("Filament proxy (high-density voxels)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    overlay = np.zeros((*density[mid].shape, 3))
    dslice = density[mid]
    norm = dslice / (dslice.max() + 1e-8)
    overlay[..., 0] = norm
    overlay[..., 1] = norm * 0.6
    overlay[..., 2] = norm * 0.3
    overlay[filament_mask[mid]] = [1.0, 0.75, 0.2]
    overlay[void_mask[mid]] = [0.2, 0.45, 1.0]
    axes[1, 1].imshow(overlay, origin="lower")
    axes[1, 1].set_title("Overlay: density + voids + filaments")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_nn_histogram(positions: np.ndarray, report: dict, path) -> None:
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=2).fit(positions)
    nn = nbrs.kneighbors(positions)[0][:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nn, bins=80, color="#818cf8", edgecolor="#312e81", alpha=0.85)
    ax.axvline(report["nearest_neighbors"]["median_nn_mpc"], color="#34d399", ls="--", lw=1.5, label="Median")
    ax.set_xlabel("Nearest-neighbor distance (Mpc)")
    ax.set_ylabel("Galaxy count")
    ax.set_title("Distribution of galaxy separations")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_dashboard(positions: np.ndarray, density: np.ndarray, report: dict, path) -> None:
    idx = subsample_indices(len(positions), 6_000)
    pts = positions[idx]
    cr = report["correlation"]
    mid = density.shape[0] // 2

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(pts[:, 0], pts[:, 1], s=0.6, c="#93c5fd", alpha=0.4, edgecolors="none")
    ax0.set_title("Spatial distribution (xy)")
    ax0.set_xlabel("x (Mpc)")
    ax0.set_ylabel("y (Mpc)")

    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(density[mid], origin="lower", cmap="inferno")
    ax1.set_title("Cosmic density field (mid slice)")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[1, 0])
    r = cr["r_mpc_centers"]
    ax2.plot(r, cr["xi_excess"], "o-", color="#34d399", ms=3)
    ax2.axhline(0, color="#6b7280", ls="--", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("r (Mpc)")
    ax2.set_ylabel("ξ excess vs random")
    ax2.set_title(f"Clustering strength @ small r: {cr['clustering_strength_small_scale']:.2f}")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    cl = report["clusters"]
    ds = report["density_structure"]
    summary = (
        f"Galaxies: {report['n_galaxies']:,}\n"
        f"Clusters: {cl['n_clusters']}\n"
        f"Clustered: {cl['fraction_clustered']*100:.1f}%\n"
        f"Mean NN dist: {report['nearest_neighbors']['mean_nn_mpc']:.2f} Mpc\n"
        f"Void voxels: {ds['void_fraction_of_occupied']*100:.1f}% of occupied\n"
        f"Filament voxels: {ds['filament_voxel_count']:,}\n"
    )
    ax3.text(0.05, 0.95, summary, va="top", fontsize=12, family="monospace", color="#e5e7eb")
    ax3.set_title("Analysis summary", loc="left")

    fig.suptitle("Cosmic Web Pattern Lab — SDSS Sample", fontsize=14, y=1.01)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_projections(positions: np.ndarray, path) -> None:
    idx = subsample_indices(len(positions), 8_000)
    p = positions[idx]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pairs = [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]
    for ax, (a, b, name) in zip(axes, pairs):
        ax.scatter(p[:, a], p[:, b], s=0.5, c="#a78bfa", alpha=0.35, edgecolors="none")
        ax.set_xlabel(f"axis {a} (Mpc)")
        ax.set_ylabel(f"axis {b} (Mpc)")
        ax.set_title(f"Projection {name}")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    positions = np.load(POSITIONS_NPY)
    density = np.load(DENSITY_NPY)
    void_mask = np.load(VOID_MASK_NPY) if VOID_MASK_NPY.exists() else None
    filament_mask = np.load(FILAMENT_MASK_NPY) if FILAMENT_MASK_NPY.exists() else None
    labels = np.load(CLUSTER_LABELS_NPY) if CLUSTER_LABELS_NPY.exists() else None

    patterns_path = RESULTS_DIR / "patterns.json"
    if not patterns_path.exists():
        raise FileNotFoundError("Run: python -m src.patterns")
    report = json.loads(patterns_path.read_text(encoding="utf-8"))

    save_dashboard(positions, density, report, RESULTS_DIR / "dashboard.png")
    save_correlation_comparison(report, RESULTS_DIR / "correlation_comparison.png")
    save_projections(positions, RESULTS_DIR / "projections_3panel.png")
    save_nn_histogram(positions, report, RESULTS_DIR / "nearest_neighbors.png")

    if void_mask is not None and filament_mask is not None:
        save_density_structure(density, void_mask, filament_mask, RESULTS_DIR / "density_structure.png")
    if labels is not None:
        save_cluster_plot(positions, labels, RESULTS_DIR / "positions_clusters.png")

    cr = report["correlation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cr["r_mpc_centers"], cr["xi_data"], "o-", ms=3, color="#60a5fa")
    ax.axhline(0, color="#6b7280", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("r (Mpc)")
    ax.set_ylabel("ξ proxy")
    ax.set_title("Galaxy two-point correlation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "correlation.png", dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved portfolio figures in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
