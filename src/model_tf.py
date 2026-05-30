from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import (
    AUTOENCODER_EPOCHS,
    BATCH_SIZE,
    DENSITY_NPY,
    FIGURE_DPI,
    LATENT_DIM,
    MODELS_DIR,
    PATCH_SIZE,
    RANDOM_SEED,
    RESULTS_DIR,
)


def extract_patches(volume: np.ndarray, patch_size: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    d = volume.shape[0]
    patches = []
    origins = []
    step = patch_size // 2
    for i in range(0, d - patch_size + 1, step):
        for j in range(0, d - patch_size + 1, step):
            for k in range(0, d - patch_size + 1, step):
                patch = volume[i : i + patch_size, j : j + patch_size, k : k + patch_size]
                if patch.max() > 0:
                    patches.append(patch)
                    origins.append((i, j, k))
    if not patches:
        raise ValueError("No non-empty patches; check density grid.")
    stacked = np.stack(patches, axis=0)[..., np.newaxis].astype(np.float32)
    return stacked, origins


def build_autoencoder(patch_size: int, latent_dim: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(patch_size, patch_size, patch_size, 1))
    x = tf.keras.layers.Conv3D(16, 3, activation="relu", padding="same")(inp)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu")(x)

    x = tf.keras.layers.Dense(4 * 4 * 4 * 32, activation="relu")(latent)
    x = tf.keras.layers.Reshape((4, 4, 4, 32))(x)
    x = tf.keras.layers.UpSampling3D(2)(x)
    x = tf.keras.layers.Conv3D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling3D(2)(x)
    out = tf.keras.layers.Conv3D(1, 3, activation="sigmoid", padding="same")(x)

    return tf.keras.Model(inp, out)


def anomaly_scores(model: tf.keras.Model, patches: np.ndarray) -> np.ndarray:
    recon = model.predict(patches, verbose=0)
    return np.mean((patches - recon) ** 2, axis=(1, 2, 3, 4))


def save_anomaly_figure(patches: np.ndarray, scores: np.ndarray, top_idx: np.ndarray, path) -> None:
    n = min(6, len(top_idx))
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5), squeeze=False)
    mid = PATCH_SIZE // 2
    for col, idx in enumerate(top_idx[:n]):
        orig = patches[idx, :, :, :, 0]
        axes[0, col].imshow(orig[mid], origin="lower", cmap="magma")
        axes[0, col].set_title(f"#{idx}\nMSE={scores[idx]:.4f}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(orig[:, mid, :], origin="lower", cmap="inferno")
        axes[1, col].axis("off")
    fig.suptitle("Highest reconstruction-error density patches", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor="#0f1117")
    plt.close(fig)


def main() -> None:
    if not DENSITY_NPY.exists():
        raise FileNotFoundError(
            f"Missing {DENSITY_NPY}. Run: python -m src.density"
        )

    tf.random.set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    volume = np.load(DENSITY_NPY)
    patches, _ = extract_patches(volume, PATCH_SIZE)

    model = build_autoencoder(PATCH_SIZE, LATENT_DIM)
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        patches,
        patches,
        epochs=AUTOENCODER_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        verbose=1,
    )

    model_path = MODELS_DIR / "density_autoencoder.keras"
    model.save(model_path)

    scores = anomaly_scores(model, patches)
    top_k = min(10, len(scores))
    top_idx = np.argsort(scores)[-top_k:][::-1]

    save_anomaly_figure(patches, scores, top_idx, RESULTS_DIR / "anomaly_patches.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("3D autoencoder training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curve.png", dpi=FIGURE_DPI)
    plt.close(fig)

    report = {
        "n_patches": int(len(patches)),
        "mean_reconstruction_mse": float(scores.mean()),
        "std_reconstruction_mse": float(scores.std()),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "top_anomaly_patch_indices": top_idx.tolist(),
        "top_anomaly_scores": scores[top_idx].tolist(),
    }
    out = RESULTS_DIR / "anomalies.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model -> {model_path}")
    print(f"Anomaly report -> {out}")
    print(f"Figures -> anomaly_patches.png, training_curve.png")


if __name__ == "__main__":
    main()
