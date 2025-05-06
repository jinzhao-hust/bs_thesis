import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

output_file = 'pyg_train_cmp.pdf'
# 数据类型分类
type_i = ['citeseer', 'cora', 'pubmed', 'ppi']
type_ii = ['PROTEINS_full', 'OVCAR-8H', 'Yeast', 'DD', 'TWITTER-Real-Graph-Partial', 'amazon0505']
type_iii = ['amazon0505', 'artist', 'com-amazon', 'soc-BlogCatalog', 'amazon0601']
type_boundaries = [len(type_i), len(type_i) + len(type_ii)]  # 分隔 Type I / II / III

def read_and_merge(ours_file, dgl_file):
    ours = pd.read_csv(ours_file)
    dgl = pd.read_csv(dgl_file)

    ours['dataset'] = ours['dataset'].str.strip().str.lower()
    dgl['dataset'] = dgl['dataset'].str.strip().str.lower()

    merged = pd.merge(ours, dgl, on='dataset', suffixes=('_ours', '_dgl'))
    merged['speedup'] = merged['Avg.Epoch (ms)_dgl'] / merged['Avg.Epoch (ms)_ours']
    return merged

# 读取所有数据
gcn = read_and_merge('gnna_gcn.csv', 'pyg_gcn.csv')
gin = read_and_merge('gnna_gin.csv', 'pyg_gin.csv')

# 按 dataset 合并以绘图
merged_all = pd.merge(gcn[['dataset', 'speedup']], gin[['dataset', 'speedup']], on='dataset', suffixes=('_gcn', '_gin'))
datasets = merged_all['dataset'].tolist()
speedup_gcn = merged_all['speedup_gcn'].tolist()
speedup_gin = merged_all['speedup_gin'].tolist()

# 输出每个数据集的加速比
print("各数据集加速比：")
for i, ds in enumerate(datasets):
    print(f"{ds}: GCN={speedup_gcn[i]:.2f}×, GIN={speedup_gin[i]:.2f}×")

# 平均加速计算函数
def type_avg(data_list, df, label):
    return df[df['dataset'].isin([d.lower() for d in data_list])][label].mean()

print("\n平均加速比：")
print(f"GCN 总平均：{gcn['speedup'].mean():.2f}×")
print(f"GIN 总平均：{gin['speedup'].mean():.2f}×")

print(f"\nGCN Type I 平均：{type_avg(type_i, gcn, 'speedup'):.2f}×")
print(f"GCN Type II 平均：{type_avg(type_ii, gcn, 'speedup'):.2f}×")
print(f"GCN Type III 平均：{type_avg(type_iii, gcn, 'speedup'):.2f}×")

print(f"\nGIN Type I 平均：{type_avg(type_i, gin, 'speedup'):.2f}×")
print(f"GIN Type II 平均：{type_avg(type_ii, gin, 'speedup'):.2f}×")
print(f"GIN Type III 平均：{type_avg(type_iii, gin, 'speedup'):.2f}×")

# 绘图
# 绘图
x = np.arange(len(datasets))
width = 0.35
y_max = 4

plt.figure(figsize=(14, 7))
bars_gcn = plt.bar(x - width/2, speedup_gcn, width, label='GCN', color='cornflowerblue')
bars_gin = plt.bar(x + width/2, speedup_gin, width, label='GIN', color='salmon')

# 添加标注（超出y轴最大值）
for i, (bar_gcn, bar_gin) in enumerate(zip(bars_gcn, bars_gin)):
    if bar_gcn.get_height() > y_max:
        plt.text(bar_gcn.get_x() + bar_gcn.get_width() / 2, y_max - 0.2,
                 f'{bar_gcn.get_height():.1f}×', ha='center', va='top', fontsize=14, rotation=90, color='black')
    if bar_gin.get_height() > y_max:
        plt.text(bar_gin.get_x() + bar_gin.get_width() / 2, y_max - 0.2,
                 f'{bar_gin.get_height():.1f}×', ha='center', va='top', fontsize=14, rotation=90, color='black')

for boundary in type_boundaries:
    plt.axvline(x=boundary - 0.5, color='red', linestyle='--', linewidth=1)

midpoints = [
    (0 + type_boundaries[0] - 1) / 2,
    (type_boundaries[0] + type_boundaries[1] - 1) / 2,
    (type_boundaries[1] + len(datasets) - 1) / 2,
]

labels = ['Type I', 'Type II', 'Type III']
y_offset = -120  # 控制文本向下偏移

for mid, label in zip(midpoints, labels):
    plt.annotate(label,
                 xy=(mid, 0), xytext=(mid, y_offset),
                 textcoords='offset points',
                 ha='center', va='top',
                 fontsize=12, fontweight='bold',
                 annotation_clip=False)


plt.xticks(x, datasets, rotation=45, ha='right', fontsize=10, fontweight='bold')    
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.ylabel('Speedup (×)')
plt.ylim(0, y_max)
# plt.title('Inference Speedup over DGL (GCN vs GIN)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# 可选保存
plt.savefig(output_file, format='pdf', bbox_inches='tight')

plt.show()
