import matplotlib.pyplot as plt
import numpy as np
import io

# Data provided by the user
datasets_data = {
    'amazon0505': (0.051, 5.756),
    'artist': (0.011, 0.778),
    'com-amazon': (0.02, 4.2678),
    'soc-BlogCatalog': (0.0196, 1.3386),
    'amazon0601': (0.083, 5.355)
}

dataset_names = ['amazon0505', 'artist', 'com-amazon', 'soc-BlogCatalog', 'amazon0601']

reorder_percentages = []
train_percentages = []

for name in dataset_names:
    r_raw, t_raw = datasets_data[name]
    total = r_raw + t_raw
    if total == 0:
        re_pct = 0
        tr_pct = 0
    else:
        re_pct = (r_raw / total) * 100
        tr_pct = (t_raw / total) * 100
    reorder_percentages.append(re_pct)
    train_percentages.append(tr_pct)

# Font settings
plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

# Colors
color_reordering = 'green'
color_training = 'orange'

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

bar_height = 0.75
y_pos = np.arange(len(dataset_names))

# Plot bars
ax.barh(y_pos, reorder_percentages, height=bar_height, label='Reordering', color=color_reordering, edgecolor='black', linewidth=1.5)
ax.barh(y_pos, train_percentages, height=bar_height, left=reorder_percentages, label='Training', color=color_training, edgecolor='black', linewidth=1.5)

# X-axis on top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xticklabels([f'{x}%' for x in np.arange(0, 101, 20)], fontweight='bold')
ax.set_xlim(0, 100)

# Y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(dataset_names, fontweight='bold')
ax.invert_yaxis()

# Legend
legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, handletextpad=0.5, columnspacing=2.0)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Grid and spines
ax.xaxis.grid(True, linestyle='-', color='gainsboro', linewidth=1)
ax.set_axisbelow(True)

ax.spines['top'].set_linewidth(2)
ax.spines['top'].set_capstyle('butt')
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_capstyle('butt')

# Tick styling
ax.tick_params(axis='x', which='major', direction='out', length=8, width=2, colors='black', pad=8)
ax.tick_params(axis='y', which='major', length=0, pad=10)

plt.tight_layout(rect=[0.05, 0.12, 0.95, 0.92])
plt.savefig('reorder_overhead.pdf', format='pdf', bbox_inches='tight')
plt.show()
