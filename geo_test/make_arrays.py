"""Build a tiny synthetic 3-zone reflectance scene for index checks."""

import os

import numpy as np

H, W = 8, 8


def make(zones):
    """Fill an HxW array from three zone values (veg, water, bare)."""
    a = np.empty((H, W), dtype=np.float32)
    a[0:3, :] = zones[0]
    a[3:5, :] = zones[1]
    a[5:8, :] = zones[2]
    return a


def main():
    os.makedirs("geo_test", exist_ok=True)
    bands = {
        "red": [0.05, 0.06, 0.25],
        "nir": [0.45, 0.02, 0.30],
        "green": [0.06, 0.05, 0.20],
        "swir22": [0.20, 0.02, 0.35],
    }
    for name, zones in bands.items():
        np.save(f"geo_test/{name}.npy", make(zones))
    print("wrote", sorted(os.listdir("geo_test")))


if __name__ == "__main__":
    main()
