import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import os

# --- Data Loading ---
# Define directory and file naming pattern
# Get the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
file_pattern = "3D_{}m_1stinc_10^6_Knu.csv"

def load_data(length):
    path = f"{base_dir}/{length}m/{file_pattern.format(length)}"
    df = pd.read_csv(path, dtype=float)
    return df['Counts'].tolist()

counts_100m = load_data(100)
counts_300m = load_data(300)
counts_1km = load_data(1000)
counts_3km = load_data(3000)
counts_10km = load_data(10000)

# --- Calculation for Axes ---
SCALE = 10000 
y_values = np.linspace(0, 70, 70 * SCALE)
# Calculate percentile values for plotting
x_percentiles = [
    np.percentile(counts_100m, 100 - y_values),
    np.percentile(counts_300m, 100 - y_values),
    np.percentile(counts_1km, 100 - y_values),
    np.percentile(counts_3km, 100 - y_values),
    np.percentile(counts_10km, 100 - y_values)
]

# --- Visualization Setup ---
# Create custom colormap
custom_colors = [
    (0.0, 0.0, 1.0), # Blue
    (0.2, 0.0, 1.0), # Blue-Purple
    (0.5, 0.0, 1.0), # Purple
    (0.5, 0.0, 0.5), # Red-Purple
    (0.8, 0.0, 0.2)  # Red
]
cmap = LinearSegmentedColormap.from_list('custom_gradient', custom_colors, N=5)
plot_colors = [cmap(i/4) for i in range(5)]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['L/D = 1', 'L/D = 3', 'L/D = 10', 'L/D = 30', 'L/D = 100']
marker_sizes = [2] * len(y_values)

# Plotting each series
for i in range(5):
    ax.scatter(
        x_percentiles[i], 
        y_values / 100, 
        s=marker_sizes, 
        color=plot_colors[i], 
        label=labels[i]
    )

# Formatting
ax.set_xlabel('Number of Adsorption Events', fontsize=20)
ax.set_ylabel('Residual Fraction', fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-6, 1)
ax.grid(True, linestyle='-', zorder=-1)

# Legend
ax.legend(
    loc='upper right', 
    fontsize=11, 
    facecolor='white', 
    edgecolor='black', 
    framealpha=1, 
    markerscale=5
)
# --- Save Plot ---
# Create a folder for output images in the script directory
output_dir = os.path.join(base_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Save the figure
output_path = os.path.join(output_dir, "adsorption_events_distribution.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300) # dpi=300 for high quality
plt.close()

print(f"Plot saved successfully at: {output_path}")