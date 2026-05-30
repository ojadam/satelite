# Cosmic Web Pattern Lab

## What is this?

I mapped around 30000 galaxies from the from the Sloan Digital Sky Survey (SDSS) into 3D moving space and tested whether their positions are random or actually make sense with modern physics and large-scale structure of cosmic web.

## What I actually did

Step by step, no jargon wall.

**1. Get galaxy data from SDSS**  
I downloaded ~**30,000 galaxies** from the Sloan Digital Sky Survey. Each one has a spot on the sky (right ascension + declination) and a **redshift** (how stretched the light is). We use redshift to guess how far away it is.

**2. Make a 3D map**  
RA and dec only tell you a direction. They do not give you a full position in space. I used normal astronomy code to turn each galaxy into **x, y, z** in megaparsecs (Mpc). A megaparsec is just a giant distance unit. After this you get a **cloud of points in a box**. That box is our little slice of the universe.

**3. Count galaxies in a grid**  
I chopped the box into **64 x 64 x 64** tiny cubes (think Minecraft). I counted how many galaxies landed in each cube. **Hot spots** = lots of galaxies. **Cold spots** = almost none. That grid is a rough map of clumps and holes. It is not a telescope photo.

**4. Check patterns vs random**  
- **Grouping:** I ran DBSCAN. It finds tight clusters of points. Random scatter would not give you this many big clumps.  
- **Voids:** Cubes with almost no galaxies = empty-ish gaps between the clumps.  
- **Correlation:** I checked if galaxy pairs sit closer than you'd get if you shuffled the same 30k points inside the same box. If real data beats random, the structure is real, not luck.

**5. ML bit**  
I cut the grid into **16 x 16 x 16** chunks and trained a **3D autoencoder** to rebuild them. Most chunks look the same (mostly empty, few bright cells). Where the model messes up the rebuild, that chunk looks **weird** compared to the rest. That is not a named discovery. It is just "this patch looks off."

### What the numbers said (my run)

- **30,000 galaxies**, redshift about **0.01 to 0.25** (close-ish in cosmic terms).  
- **170 groups** from DBSCAN. About **53%** of galaxies landed in some group. The rest were loners or in-between.  
- The **two biggest groups** had ~**4,200** and ~**3,900** galaxies. That is fat clumps in this sample.  
- **Median gap to your nearest neighbor: ~2 Mpc.** Lots of galaxies sit way closer to another one than you'd expect from random sprinkling.  
- **Vs a random catalog in the same box:** clustering score was about **1.6x** higher at small scales. Plain English: galaxies like having neighbors nearby more than random points do.


## Results (plots)

All plots are in `results/` in the repo. They display on GitHub in this file.

### Overview

![Project dashboard](results/dashboard.png)

### Are galaxies clumped vs random?

**Blue** = real SDSS data. **Red** = the same number of random points in the same box. **Green** = the difference. On small separations, green sits above zero. The real sample is more clustered than random.

![Correlation: data vs random](results/correlation_comparison.png)

### Where the groups are

Each color is one DBSCAN cluster. **Gray** = no cluster assigned. The map shows large clusters, not a uniform spread.

![Galaxy positions by cluster](results/positions_clusters.png)

### Density, voids, and filaments

**Top row:** galaxy count per voxel. **Bottom row:** low-density void regions (blue) and high-density filament regions (orange).

![Density structure](results/density_structure.png)

### How far apart galaxies sit

Histogram of distance to the nearest neighbor. The left-side peak shows many galaxies have a close neighbor. The spacing is not random.

![Nearest-neighbor distances](results/nearest_neighbors.png)

### Three views of the same cloud

The same **30,000** points in **xy**, **xz**, and **yz** projection. Three angles on one cloud.

![3D projections](results/projections_3panel.png)

### ML: high-error density patches

These **16 x 16 x 16** chunks have the highest autoencoder reconstruction error in the run. High error = the patch differs from most other patches in the grid.

![Anomaly patches](results/anomaly_patches.png)

![Autoencoder training loss](results/training_curve.png)

## Run it

```bash
python -m venv .venv
# windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py all
```

The first run downloads from SDSS and writes `data/galaxies.csv`. Later runs use that file:

```bash
python run.py patterns
python run.py viz
python run.py analyze
```

## Stack

Python, TensorFlow, Astropy, astroquery, scikit-learn, matplotlib.

## Caveats (if you know astronomy)

The grid is **64 x 64 x 64** only. The autoencoder has no labels. High-error patches are not discoveries. They are patches that differ from the average density pattern.

Data: [SDSS](https://www.sdss.org/).
