import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io
import os # Optional: to manage output directory
import numpy as np # Import numpy for numerical operations

# --- 1. Data Preparation ---
# Store the CSV data provided in a multiline string

# dim worker研究
csv_data = """dataset, 1, 2, 4, 8, 16, 32
amazon0505, 27.455, 26.267, 27.218, 26.505, 27.088, 26.310
artist, 3.660, 3.764, 3.628, 3.854, 3.614, 3.917
com-amazon, 18.847, 19.789, 19.359, 19.434, 19.522, 19.628
soc-BlogCatalog, 6.023, 6.396, 6.104, 6.403, 6.063, 6.479
amazon0601, 26.009, 26.472, 26.327, 25.831, 25.676, 26.200
"""
# ngs研究
# csv_data = """dataset, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
# amazon0505, 26.452, 26.319, 26.235, 26.377, 26.366, 26.187, 27.150, 26.651, 26.796, 26.924
# artist, 3.692, 3.581, 3.536, 3.648, 3.637, 3.618, 3.824, 3.803, 3.616, 4.044
# com-amazon, 18.834, 18.820, 18.809, 18.979, 19.023, 18.986, 19.533, 19.210, 19.408, 21.199
# soc-BlogCatalog, 6.214, 6.032, 6.065, 6.220, 6.124, 6.222, 6.467, 6.330, 6.260, 7.050
# amazon0601, 25.774, 25.723, 25.729, 25.758, 25.639, 25.631, 26.216, 26.090, 26.273, 28.647
# """


# Use io.StringIO to simulate reading from a file
data_io = io.StringIO(csv_data)

# Read the data using pandas
df = pd.read_csv(data_io)

# --- 2. Define X-axis Labels and Plotting Indices ---
# These are the actual labels we want on the x-axis
x_labels = df.columns[1:]
num_points = len(x_labels)
# These are the equally spaced positions we'll use for plotting
x_indices = np.arange(num_points) # Creates [0, 1, 2, ..., num_points-1]

# --- 3. Normalize Data ---
# Create a copy for normalized data
df_normalized = df.copy()

# Normalize each row based on its value at the first data point (column index 1)
baseline_column = df.columns[1] # Name of the first data column ('1')
for col in x_labels: # Iterate using the label names
    # Convert column to numeric
    df_normalized[col] = pd.to_numeric(df_normalized[col])
    # Normalize: (value / value_at_baseline) * 100
    df_normalized[col] = (df_normalized[col] / df.iloc[:, 1]) * 100 # df.iloc[:, 1] is the baseline column

# --- 4. Define Plot Styles ---
# Match colors and markers from the image
styles = {
    'amazon0505': {'color': 'blue', 'marker': 'D', 'label': 'amazon0505'}, # Diamond
    'artist': {'color': 'red', 'marker': 's', 'label': 'artist'}, # Square
    'com-amazon': {'color': 'goldenrod', 'marker': '^', 'label': 'com-amazon'}, # Triangle up
    'soc-BlogCatalog': {'color': 'limegreen', 'marker': 'X', 'label': 'soc-BlogCatalog'}, # Cross (filled)
    'amazon0601': {'color': 'mediumturquoise', 'marker': 'o', 'label': 'amazon0601'} # Circle
}

# Global plot style settings
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 14 # Adjust base font size if needed

# --- 5. Create Plot ---
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

# Plot each dataset using indices for x-position
for index, row in df_normalized.iterrows():
    dataset_name = row['dataset']
    if dataset_name in styles:
        style = styles[dataset_name]
        # Plot using x_indices for position and row[x_labels] for y-values
        ax.plot(
            x_indices, # Use the equally spaced indices [0, 1, 2...] for plotting
            row[x_labels].values.astype(float), # Ensure data is float
            label=style['label'],
            color=style['color'],
            marker=style['marker'],
            markersize=8, # Adjust marker size
            linewidth=2.5, # Adjust line width
            linestyle='-'
        )
    else:
        # Default style if name not found (optional)
        ax.plot(
            x_indices,
            row[x_labels].values.astype(float),
            label=dataset_name
            )

# --- 6. Customize Plot Appearance ---
# Y-axis (MODIFIED)
ax.set_ylabel('Norm. Runtime (%)', fontsize=16, weight='bold')
ax.set_ylim(0, 120) # Set upper limit to 120%
ax.set_yticks(range(0, 121, 20)) # Set ticks from 0 to 120, every 20
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))

# X-axis (MODIFIED for alignment)
ax.set_xlabel('')
ax.set_xticks(x_indices) # Set tick positions to the equally spaced indices
ax.set_xticklabels(x_labels) # Set the tick labels to the original values ('1', '2', '4'...)
# Explicitly set x-axis limits to ensure '1' (at index 0) is at the origin
# Add slight padding to avoid points hitting the exact edge of the plot area
ax.set_xlim(left=-0.2, right=num_points - 1 + 0.2)


# Legend (same as before)
legend = ax.legend(fontsize=16, frameon=True, facecolor='white', framealpha=0.8)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Spine thickness (plot border - same as before)
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Tick parameters (same as before)
ax.tick_params(axis='both', which='major', direction='in', width=2, length=6, labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Remove minor ticks if they appear automatically and look cluttered
ax.minorticks_off()

# --- 7. Save and Show ---
output_filename = 'runtime_plot_equal_spacing_ylim120.pdf' # Changed filename
plt.savefig(output_filename, bbox_inches='tight', format='pdf')
print(f"Plot saved as {output_filename}")

plt.show() # Display the plot