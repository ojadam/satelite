from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from src.config import (
    CATALOG_CSV,
    CLUSTER_LABELS_NPY,
    DENSITY_NPY,
    MAX_GALAXIES,
    POSITIONS_NPY,
    RESULTS_DIR,
    Z_MAX,
    Z_MIN,
)


def interpret_clustering(strength: float) -> str:
    if strength > 2.0:
        return "Strong small-scale clustering detected versus a uniform random catalog."
    if strength > 0.5:
        return "Moderate clustering: galaxies preferentially sit closer than a random placement."
    return "Weak clustering signal; consider more galaxies or tighter correlation bins."


def build_report_md(data: dict, generated_at: str) -> str:
    g = data["geometry"]
    nn = data["nearest_neighbors"]
    ds = data["density_structure"]
    cl = data["clusters"]
    cr = data["correlation"]
    strength = cr["clustering_strength_small_scale"]
    interp = interpret_clustering(strength)

    lines = [
        "# Cosmic Web Analysis Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Dataset",
        f"- Source: SDSS spectroscopic sample",
        f"- Galaxies: **{data['n_galaxies']:,}** (limit {MAX_GALAXIES:,})",
        f"- Redshift slice: **{Z_MIN} – {Z_MAX}**",
        f"- Catalog: `{CATALOG_CSV.name}`",
        "",
        "## Volume",
        f"- Box span (Mpc): {g['box_span_mpc'][0]:.0f} × {g['box_span_mpc'][1]:.0f} × {g['box_span_mpc'][2]:.0f}",
        f"- Volume: {g['box_volume_mpc3']:.3e} Mpc³",
        f"- Number density: {g['number_density_per_mpc3']:.4e} galaxies / Mpc³",
        "",
        "## Nearest neighbors",
        f"- Mean separation: **{nn['mean_nn_mpc']:.2f} Mpc**",
        f"- Median: {nn['median_nn_mpc']:.2f} Mpc (σ = {nn['std_nn_mpc']:.2f})",
        f"- 10th–90th percentile: {nn['p10_nn_mpc']:.2f} – {nn['p90_nn_mpc']:.2f} Mpc",
        "",
        "## Clustering (DBSCAN)",
        f"- Clusters found: **{cl['n_clusters']}**",
        f"- Galaxies in clusters: {cl['fraction_clustered']*100:.1f}%",
        f"- Largest cluster: {cl['largest_cluster_fraction']*100:.1f}% of sample",
        f"- Top cluster sizes: {cl['cluster_sizes_top10'][:5]}",
        "",
        "## Density field (64³ voxels)",
        f"- Void-like voxels: {ds['void_fraction_of_occupied']*100:.1f}% of occupied cells",
        f"- Filament-like voxels (top density): {ds['filament_fraction_of_grid']*100:.2f}% of grid",
        f"- Mean voxel density: {ds['mean_occupancy_density']:.4f}",
        "",
        "## Two-point correlation",
        f"- Small-scale excess vs random: **{strength:.3f}**",
        f"- {interp}",
        "",
        "## Figures",
        "| File | Description |",
        "|------|-------------|",
        "| `dashboard.png` | Overview: positions, density, correlation |",
        "| `positions_clusters.png` | DBSCAN-colored projection |",
        "| `density_structure.png` | Slices + void/filament masks |",
        "| `correlation_comparison.png` | Data vs random Poisson cube |",
        "| `nearest_neighbors.png` | Separation distribution |",
        "| `anomaly_patches.png` | TensorFlow high-error regions |",
        "",
        "## Conclusion",
        "Galaxy positions in this volume show **non-random large-scale structure**:",
        "overdense clusters, underdense void proxies, and elevated pair counts at small separations",
        "compared to a uniform random catalog in the same bounding box.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    patterns_path = RESULTS_DIR / "patterns.json"
    if not patterns_path.exists():
        raise FileNotFoundError("Run: python -m src.patterns")

    patterns = json.loads(patterns_path.read_text(encoding="utf-8"))
    anomalies_path = RESULTS_DIR / "anomalies.json"
    anomalies = {}
    if anomalies_path.exists():
        anomalies = json.loads(anomalies_path.read_text(encoding="utf-8"))

    labels = None
    if CLUSTER_LABELS_NPY.exists():
        labels = np.load(CLUSTER_LABELS_NPY)

    full = {
        **patterns,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project": "Cosmic Web Pattern Lab",
        },
        "tensor_flow": anomalies,
        "cluster_label_stats": {
            "n_unique_labels": int(len(np.unique(labels))) if labels is not None else 0,
        },
    }

    full_path = RESULTS_DIR / "full_analysis.json"
    full_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    generated_at = full["meta"]["generated_at"]
    md = build_report_md(patterns, generated_at)
    md_path = RESULTS_DIR / "REPORT.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"Full analysis -> {full_path}")
    print(f"Report -> {md_path}")


if __name__ == "__main__":
    main()
