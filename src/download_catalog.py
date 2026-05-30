from __future__ import annotations

import sys

import pandas as pd

from src.config import CATALOG_CSV, DATA_DIR, MAX_GALAXIES, Z_MAX, Z_MIN


def fetch_sdss_galaxies() -> pd.DataFrame:
    from astroquery.sdss import SDSS

    query = f"""
    SELECT TOP {MAX_GALAXIES}
        p.ra, p.dec, s.z AS redshift
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestObjID = p.objID
    WHERE s.z BETWEEN {Z_MIN} AND {Z_MAX}
      AND s.zWarning = 0
    """
    print("Querying SDSS (may take 1–3 minutes)...")
    result = SDSS.query_sql(query)
    df = result.to_pandas()
    df = df.dropna()
    df = df.rename(columns={"ra": "ra_deg", "dec": "dec_deg"})
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CATALOG_CSV.exists():
        print(f"Catalog already exists: {CATALOG_CSV}")
        df = pd.read_csv(CATALOG_CSV)
        print(f"  {len(df):,} galaxies loaded from disk.")
        return

    try:
        df = fetch_sdss_galaxies()
    except Exception as exc:
        print("SDSS download failed:", exc, file=sys.stderr)
        print(
            "\nTips: check internet, try again later, or place your own CSV at:\n"
            f"  {CATALOG_CSV}\n"
            "Required columns: ra_deg, dec_deg, redshift"
        )
        sys.exit(1)

    df.to_csv(CATALOG_CSV, index=False)
    print(f"Saved {len(df):,} galaxies -> {CATALOG_CSV}")


if __name__ == "__main__":
    main()
