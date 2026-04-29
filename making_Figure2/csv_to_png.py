import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Parameters (consistent with simulation code)
CAVE_LENGTH = 100
CAVE_WIDTH = 100
CAVE_HEIGHT = 60
NUM_MOLECULES = 10000
TOTAL_SIMULATIONS = 100  
TOTAL_MOLECULES = NUM_MOLECULES * TOTAL_SIMULATIONS
BIN_WIDTH = CAVE_LENGTH / 20  # Equivalent to l / 20

def get_bin_areas_circle(radius, bins):
    delta_x = (2 * radius) / bins
    areas = []
    for i in range(bins):
        x_lower = -radius + i * delta_x
        x_mid = x_lower + (delta_x / 2)
        half_width = np.sqrt(radius**2 - x_mid**2)
        
        area_circle = 2 * half_width * delta_x
        area_ex_circle = 2 * (BIN_WIDTH * CAVE_HEIGHT + BIN_WIDTH * CAVE_WIDTH) - area_circle
        areas.append(area_ex_circle)
    return areas

# Surface area definitions
s = 2 * (BIN_WIDTH * CAVE_HEIGHT + BIN_WIDTH * CAVE_WIDTH)
s1 = 2 * (BIN_WIDTH * CAVE_HEIGHT + BIN_WIDTH * CAVE_WIDTH) + CAVE_HEIGHT * CAVE_WIDTH
s2 = 2 * (BIN_WIDTH * CAVE_HEIGHT + BIN_WIDTH * CAVE_WIDTH) - ((CAVE_WIDTH / 2) ** 2 * np.pi) / 2

bin_edges = np.arange(-CAVE_LENGTH, CAVE_LENGTH + 1, BIN_WIDTH)
num_bins = len(bin_edges) - 1
divs = np.full(num_bins, s, dtype=float)

if num_bins > 0:
    divs[0] = s1
    divs[-1] = s1
    center_idx = num_bins // 2
    # Update central bins with circular area adjustments
    areas = get_bin_areas_circle(CAVE_WIDTH / 2, int(CAVE_LENGTH / BIN_WIDTH))
    divs[center_idx - 10 : center_idx + 10] = areas

# --- Main process ---
if __name__ == "__main__":
    # Get directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    num = int(np.log10(NUM_MOLECULES) + 2)
    output_path = os.path.join(base_dir, f"3D_{CAVE_LENGTH}m_1stinc_10^{num}_Knu.csv")

    print(f"Reading data from '{output_path}' and generating histograms...")

    df = pd.read_csv(output_path)
    max_count = int(df["adsorption_count"].max())
    threshold_n = None


    # Determine threshold_n
    for idx in range(1, max_count + 1):
        x_data = df[df["adsorption_count"] == idx]["adsorption_x"].dropna().values
        if len(x_data) == 0: continue

        hist, _ = np.histogram(x_data, bins=bin_edges)
        norm_hist = hist / divs
        uniform_density = len(x_data) / (2 * (CAVE_HEIGHT * CAVE_WIDTH + CAVE_HEIGHT * 2 * CAVE_LENGTH + CAVE_WIDTH * 2 * CAVE_LENGTH))

        if threshold_n is None and norm_hist[0] > uniform_density and norm_hist[-1] > uniform_density:
            threshold_n = idx
            print(f"Threshold reached at n={threshold_n}")
            break

    # Generate plots
    for idx in [1, round(0.5 * threshold_n), threshold_n, 5 * threshold_n, 10 * threshold_n]:
        x_data = df[df["adsorption_count"] == idx]["adsorption_x"].dropna().values
        if len(x_data) == 0: continue

        hist, _ = np.histogram(x_data, bins=bin_edges)
        norm_hist = hist / divs
        uniform_density = len(x_data) / (2 * (CAVE_HEIGHT * CAVE_WIDTH + CAVE_HEIGHT * 2 * CAVE_LENGTH + CAVE_WIDTH * 2 * CAVE_LENGTH))

        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], norm_hist, width=np.diff(bin_edges), color="black", edgecolor="white", align="edge")
        plt.xlabel("Distance into cave [m]", fontsize=33)
        plt.ylabel("Surface Density [/m$^2$]", fontsize=33)
        plt.yscale("log")
        plt.ylim(1 / s1, TOTAL_MOLECULES / s1)
        plt.axhline(y=uniform_density, color="red", linestyle=":", linewidth=10)
        plt.xticks([-CAVE_LENGTH, -CAVE_LENGTH/2, 0, CAVE_LENGTH/2, CAVE_LENGTH])
        plt.tick_params(axis="x", labelsize=35)
        plt.tick_params(axis="y", labelsize=35)
        plt.text(CAVE_LENGTH, TOTAL_MOLECULES / s1 - TOTAL_MOLECULES / (5 * s1), 
                 f"{len(x_data)/TOTAL_MOLECULES:.1%}", va="top", ha="right", fontsize=50)
        plt.text(-CAVE_LENGTH, TOTAL_MOLECULES / s1 - TOTAL_MOLECULES / (5 * s1), 
                 f"n={idx}", va="top", ha="left", fontsize=50)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"adsorption_hist_{idx}.png"))
        plt.close()

    print(f"Histograms saved in '{base_dir}'.")
