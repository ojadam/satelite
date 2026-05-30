# Cosmic Web Pattern Lab

## What is this?

I pulled ~30k galaxies from SDSS (real survey data, not made up), turned them into a 3D map, and checked if they're scattered randomly or clumped up like people say the "cosmic web" should look.

## What I actually did

Here is the pipeline in plain terms, step by step.

**1. Get real galaxy data from SDSS**  
I downloaded about 30,000 galaxies from the Sloan Digital Sky Survey. For each one I have where it sits on the sky (right ascension and declination) and how far away it is (redshift). Redshift is basically “how much the light is stretched” — we use it to estimate distance.

**2. Turn sky coordinates into a 3D map**  
RA and dec alone are directions, not a full position in space. I used standard astronomy libraries to convert each galaxy into x, y, z in megaparsecs (Mpc). One megaparsec is a huge distance unit astronomers use. After this step you can imagine a cloud of points floating in a box — that is our chunk of the universe.

**3. Build a simple density grid**  
I divided that 3D box into 64 × 64 × 64 small cubes (like Minecraft blocks). I counted how many galaxies fell into each cube. Bright / hot spots on the grid mean “lots of galaxies here.” Dark spots mean “almost empty.” That grid is a rough picture of structure — clumps and gaps — not a photo from a telescope.

**4. Look for patterns and compare to random**  
- **Grouping:** I ran DBSCAN, which finds groups of points that sit close together. If galaxies were random, you would not see so many big clumps.  
- **Voids:** I marked cubes with very few galaxies as “void-like” — empty-ish regions between the clumps.  
- **Correlation:** I measured whether galaxy pairs tend to be closer than you would expect if you shuffled the same 30,000 points randomly inside the same box. If the real data wins, structure is real, not luck.

**5. Small machine-learning piece**  
I cut the density grid into little 16 × 16 × 16 cubes and trained a 3D autoencoder to copy them back. Most cubes look similar (sparse, a few bright voxels). Where the model reconstructs badly, that cube is a bit unusual compared to the rest. It is not a labeled “discovery” — just “this patch looks different.”

### What the numbers said (on my run)

- **30,000 galaxies**, distances roughly between redshift 0.01 and 0.25 (relatively nearby in cosmic terms).  
- **170 groups** found by DBSCAN; about **53%** of galaxies belonged to some group (the rest were loners or in-between).  
- The **two biggest groups** had about **4,200** and **3,900** galaxies — clear mega-clumps in this sample.  
- **Median distance to the nearest other galaxy: ~2 Mpc** — many galaxies have a neighbor much closer than the average spacing would suggest if they were sprinkled at random.  
- **Versus a random catalog in the same box:** at small scales the clustering score was about **~1.6 higher** than random. In simple words: galaxies like having neighbors nearby more than random points do.


## Results (plots)

All figures are in `results/` if you clone the repo. They also show up here on GitHub — no need to open files one by one.

### Overview

![Project dashboard](results/dashboard.png)

### Are galaxies clumped vs random?

Blue = real SDSS sample. Red = same number of points placed randomly in the same box. Green = the difference. If structure is real, the green line should sit above zero on the left (small separations).

![Correlation: data vs random](results/correlation_comparison.png)

### Where the groups are

Each color is a DBSCAN cluster. Gray = not assigned to a cluster. You can see big clumps instead of an even sprinkle.

![Galaxy positions by cluster](results/positions_clusters.png)

### Density, voids, and filaments

Top row: how many galaxies per voxel. Bottom row: low-density “void” areas (blue) and high-density “filament” areas (orange) on top of the map.

![Density structure](results/density_structure.png)

### How far apart galaxies sit

Histogram of “distance to your closest neighbor.” The spike on the left means many galaxies have a nearby buddy — another sign they are not random.

![Nearest-neighbor distances](results/nearest_neighbors.png)

### Three views of the same cloud

Same 30k points seen from xy, xz, and yz — helps see depth and filaments.

![3D projections](results/projections_3panel.png)

### ML: unusual density patches

The autoencoder struggled most on these small 3D chunks (higher reconstruction error = less “typical” for this dataset).

![Anomaly patches](results/anomaly_patches.png)

![Autoencoder training loss](results/training_curve.png)

## Run it

```bash
python -m venv .venv
# windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py all
```

First run hits SDSS over the network and saves `data/galaxies.csv`. After that you can redo analysis without re-downloading:

```bash
python run.py patterns
python run.py viz
python run.py analyze
```

## Stack

Python, TensorFlow, Astropy, astroquery, scikit-learn, matplotlib.

## Caveats (read this if you're an astronomer)

This is a portfolio / learning project. The correlation estimator is simplified, the grid is only 64³, and the autoencoder has no labels — "anomalies" just mean atypical patches, not discoveries.

Data: [SDSS](https://www.sdss.org/).
