from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18

from src.config import CATALOG_CSV, DATA_DIR, POSITIONS_NPY


def catalog_to_comoving_mpc(df: pd.DataFrame) -> np.ndarray:
    coord = SkyCoord(
        ra=df["ra_deg"].values * u.deg,
        dec=df["dec_deg"].values * u.deg,
        distance=Planck18.comoving_distance(df["redshift"].values),
    )
    cart = coord.cartesian.xyz.to(u.Mpc).value.T
    cart = cart - cart.mean(axis=0)
    return cart.astype(np.float32)


def main() -> None:
    if not CATALOG_CSV.exists():
        raise FileNotFoundError(
            f"Missing {CATALOG_CSV}. Run: python -m src.download_catalog"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CATALOG_CSV)
    positions = catalog_to_comoving_mpc(df)
    np.save(POSITIONS_NPY, positions)

    span = positions.max(axis=0) - positions.min(axis=0)
    print(f"Saved {len(positions):,} positions -> {POSITIONS_NPY}")
    print(f"  Box span (Mpc): x={span[0]:.1f}, y={span[1]:.1f}, z={span[2]:.1f}")


if __name__ == "__main__":
    main()
