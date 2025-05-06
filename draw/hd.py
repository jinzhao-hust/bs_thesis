import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys

# --- Configuration ---
# <<< SET THE PATH TO YOUR CSV FILE HERE >>>
csv_file_path = 'study_hiddenDimension.csv' # Make sure this file exists and contains the correct data
# Set the desired output PDF filename
output_filename = 'hidden_dim_runtime.pdf'
# --- End Configuration ---


# --- 1. Load Data from CSV File ---
if not os.path.exists(csv_file_path):
    print(f"Error: CSV file not found at '{csv_file_path}'")
    sys.exit(1)

try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from '{csv_file_path}'")
except Exception as e:
    print(f"Error reading CSV file '{csv_file_path}': {e}")
    sys.exit(1)

# --- 2. Define X-axis Labels and Plotting Indices ---
try:
    # These are the actual labels we want on the x-axis (e.g., '16', '32', ...)
    x_labels = df.columns[1:]
    num_points = len(x_labels)
    # These are the equally spaced positions we'll use for plotting
    x_indices = np.arange(num_points) # Creates [0, 1, 2, ..., num_points-1]
except (IndexError, ValueError) as e:
    print(f"Error processing column headers as x-values: {e}")
    print("Ensure the CSV file has a 'dataset' column followed by numeric columns (16, 32, ...).")
    sys.exit(1)

# --- 3. Define Plot Styles (Matching the Second Image) ---
# Using slightly different color names for better match if needed
styles = {
    'amazon0505': {'color': 'royalblue', 'marker': 'D', 'label': 'amazon0505'}, # Diamond
    'artist': {'color': 'tomato', 'marker': 's', 'label': 'artist'}, # Square
    'com-amazon': {'color': 'goldenrod', 'marker': '^', 'label': 'com-amazon'}, # Triangle up
    'soc-BlogCatalog': {'color': 'darkorange', 'marker': '*', 'label': 'soc-BlogCatalog'}, # Asterisk/Star
    'amazon0601': {'color': 'lightseagreen', 'marker': 'o', 'label': 'amazon0601'} # Circle
}

# Global plot style settings
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16 # Slightly larger font for labels/ticks might match better

# --- 4. Create Plot ---
fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

# Plot each dataset using indices for x-position and original y-values
if 'dataset' not in df.columns:
     print("Error: 'dataset' column not found in the CSV file.")
     sys.exit(1)

for index, row in df.iterrows():
    dataset_name = row['dataset']
    # Select only the columns that correspond to x_labels for plotting
    plot_data = row[x_labels].values

    # Ensure data is numeric before plotting
    try:
        plot_data_numeric = plot_data.astype(float)
    except ValueError:
         print(f"Warning: Non-numeric data found for dataset '{dataset_name}' in columns {x_labels}. Skipping plot.")
         continue

    if dataset_name in styles:
        style = styles[dataset_name]
        ax.plot(
            x_indices, # Use the equally spaced indices [0, 1, 2...] for plotting
            plot_data_numeric,
            label=style['label'],
            color=style['color'],
            marker=style['marker'],
            markersize=10, # Increase marker size
            linewidth=3, # Increase line width
            linestyle='-'
        )
    else:
        # Default style if name not found (optional)
        print(f"Warning: Style not defined for dataset '{dataset_name}'. Using default style.")
        ax.plot(x_indices, plot_data_numeric, label=dataset_name, marker='o', markersize=10, linewidth=3)


# --- 5. Customize Plot Appearance ---
# Y-axis (Logarithmic)
ax.set_ylabel('Runtime (ms)', fontsize=18, weight='bold') # Larger font
ax.set_yscale('log') # Set Y-axis to logarithmic scale
ax.set_ylim(1, 1000) # Set Y limits based on image (adjust if your data differs significantly)
# Set explicit ticks for log scale
ax.set_yticks([1, 10, 100, 1000])
# Use ScalarFormatter to ensure labels are 1, 10, 100, 1000 not 10^0, 10^1 etc.
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False) # Prevent potential scientific notation
ax.yaxis.get_major_formatter().set_useOffset(False) # Prevent potential offsets


# X-axis
ax.set_xlabel('Hidden Dimension', fontsize=18, weight='bold') # Added label, larger font
ax.set_xticks(x_indices) # Set tick positions to the equally spaced indices
ax.set_xticklabels(x_labels) # Set the tick labels to the original values ('16', '32'...)
# Set x-axis limits for padding
ax.set_xlim(left=x_indices[0] - 0.2, right=x_indices[-1] + 0.2)

# Legend
legend = ax.legend(fontsize=16, frameon=True, facecolor='white', framealpha=0.8)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Spine thickness (plot border)
for spine in ax.spines.values():
    spine.set_linewidth(2.5) # Make axes lines thicker

# Tick parameters
ax.tick_params(axis='both', which='major', direction='in', width=2.5, length=7, labelsize=16) # Thicker, longer ticks
ax.tick_params(axis='y', which='minor', direction='in', width=1.5, length=4) # Add minor ticks for log scale
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')


# --- 6. Save and Show ---
try:
    plt.savefig(output_filename, bbox_inches='tight', format='pdf')
    print(f"Plot saved as {output_filename}")
except Exception as e:
    print(f"Error saving plot to '{output_filename}': {e}")

# Display the plot (optional)
plt.show()