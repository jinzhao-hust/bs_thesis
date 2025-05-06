import matplotlib.pyplot as plt
import numpy as np

# 数据集标签
datasets = ["amazon0505", "artist", "com-amazon", "soc-BlogCatalog", "amazon0601"]

# 假设的加速比（可自行修改）
speedup_gcn = [1.14, 1.01, 1.08, 1.074, 1.117]
speedup_gin = [1.093, 1.052, 1.122, 1.14, 1.082]

x = np.arange(len(datasets))  # x轴的位置
width = 0.35  # 柱状图的宽度

# 字体加粗+加大
plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, speedup_gcn, width, label='GCN', color='#1f77b4', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width / 2, speedup_gin, width, label='GIN', color='#ff7f0e', edgecolor='black', linewidth=1.2)

# 设置 Y 轴和 X 轴
ax.set_ylabel('Normalized Speedup (×)', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=15, fontweight='bold')
ax.set_ylim(0, max(max(speedup_gcn), max(speedup_gin)) + 0.5)
ax.legend()

# 显示柱状图顶部数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

# autolabel(bars1)
# autolabel(bars2)

# 边框线优化
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# 坐标轴刻度优化
ax.tick_params(axis='x', which='major', length=6, width=2)
ax.tick_params(axis='y', which='major', length=6, width=2)

plt.tight_layout()
plt.savefig('reorder_speedup.pdf', format='pdf', bbox_inches='tight')
plt.show()
