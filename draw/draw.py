import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置参数 ---
# CSV 文件路径 (请根据你的实际文件名修改)
my_operator_csv = './GCN_infer_bench/gnna_gcn.csv'
framework1_csv = './GCN_infer_bench/dgl_gcn.csv'
framework2_csv = './GCN_infer_bench/pyg_gcn.csv'
output_pdf_file = 'gcn_infer_compare.pdf'

# 对比框架的名称 (用于图例)
framework1_name = 'DGL'
framework2_name = 'PyG'
my_operator_name = 'GNNA' # 基准线，实际不绘制，用于计算

# Y轴（加速比）的最大显示值
y_axis_limit = 4
# 超过最大值时，标注文字距离柱顶的高度偏移量
annotation_offset = 0.05

# 图表标题和轴标签
chart_title = 'Speedup Comparison: My Operator vs Frameworks'
x_axis_label = 'Dataset'
y_axis_label = f'Speedup (Baseline Time / {my_operator_name} Time)'

# 柱状图颜色
color_f1 = 'tab:blue'
color_f2 = 'tab:orange'

# 设置全局字体大小和粗细
plt.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['font.weight'] = 'bold'  # 字体粗细


# --- 数据集分类和顺序 (根据提供的表格) ---
# 使用 CSV 文件中实际存在的名字！请仔细核对！
# 例如，表格是 TWITTER-Partial，你的CSV可能是 TWITTER-Real-Graph-Partial
dataset_categories = {
    'Type I': ['citeseer', 'cora', 'pubmed', 'ppi'],
    'Type II': ['PROTEINS_full', 'OVCAR-8H', 'Yeast', 'DD', 'TWITTER-Real-Graph-Partial', 'SW-620H'], # 检查 TWITTER 名称是否匹配你的CSV
    'Type III': ['amazon0505', 'artist', 'com-amazon', 'soc-BlogCatalog', 'amazon0601']
}
# 定义类型的顺序
category_order = ['Type I', 'Type II', 'Type III']

# 从分类中获取总的有序数据集列表
# **重要**: 确保这里的名字和你的 CSV 文件中的 'dataset' 列完全一致 (大小写, 特殊字符等)
ordered_datasets_from_table = [ds for cat in category_order for ds in dataset_categories[cat]]

# X轴下方分类标签的垂直偏移量（负值表示向下）
y_offset_for_type_labels = -0.35 # 可能需要根据字体大小和旋转角度调整

# --- 数据加载与处理 ---

def load_data(filepath):
    """加载CSV文件并检查必要的列"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"错误：找不到文件 '{filepath}'")
    try:
        df = pd.read_csv(filepath)
        # 规范化列名以进行检查 (转小写，去空格)
        df.columns = [col.lower().strip() for col in df.columns]

        dataset_col_name = 'dataset'
        time_col_name = 'avg.epoch (ms)'

        if dataset_col_name not in df.columns:
             raise ValueError(f"错误：文件 '{filepath}' 中缺少 '{dataset_col_name}' 列")
        if time_col_name not in df.columns:
             raise ValueError(f"错误：文件 '{filepath}' 中缺少 '{time_col_name}' 列")

        # 清理数据集名称可能的空格
        df[dataset_col_name] = df[dataset_col_name].astype(str).str.strip()
        # 尝试处理可能存在的逗号（如 '1,234.5'）
        if df[time_col_name].dtype == 'object':
             df[time_col_name] = df[time_col_name].str.replace(',', '', regex=False)
        # 转换时间为数值型
        df[time_col_name] = pd.to_numeric(df[time_col_name], errors='coerce')

        # 选择并重命名列，删除无效行
        df = df[[dataset_col_name, time_col_name]].copy()
        df.dropna(inplace=True)

        # 处理时间为0或负数的情况，避免除零错误，将其设置为一个极小值
        df.loc[df[time_col_name] <= 0, time_col_name] = 1e-9

        return df

    except Exception as e:
        print(f"加载或处理文件 '{filepath}' 时出错: {e}")
        raise

try:
    my_op_df = load_data(my_operator_csv).rename(columns={'avg.epoch (ms)': 'time_my'})
    fw1_df = load_data(framework1_csv).rename(columns={'avg.epoch (ms)': 'time_fw1'})
    fw2_df = load_data(framework2_csv).rename(columns={'avg.epoch (ms)': 'time_fw2'})

    # 合并数据：使用 'inner' 合并，确保只比较所有三个文件都包含的数据集
    merged_df = pd.merge(my_op_df, fw1_df, on='dataset', how='inner')
    merged_df = pd.merge(merged_df, fw2_df, on='dataset', how='inner')

    # --- 数据排序和过滤 ---
    # 1. 仅保留在预定义顺序列表中的数据集
    merged_df = merged_df[merged_df['dataset'].isin(ordered_datasets_from_table)]

    # 2. 按照预定义的顺序对 DataFrame 进行排序
    #    将 'dataset' 设置为索引，然后使用 reindex 按所需顺序排列
    merged_df = merged_df.set_index('dataset').reindex(ordered_datasets_from_table).reset_index()

    # 3. 检查是否有数据集在合并后丢失（如果在 reindex 后出现 NaN）
    if merged_df['time_my'].isnull().any():
        missing_in_merged = merged_df[merged_df['time_my'].isnull()]['dataset'].tolist()
        print(f"警告：以下定义在表格中的数据集在合并后的数据中缺失，将不会被绘制：{missing_in_merged}")
        merged_df.dropna(subset=['time_my', 'time_fw1', 'time_fw2'], inplace=True) # 删除这些行

    # 检查是否还有剩余数据用于绘图
    if merged_df.empty:
        print("错误：根据预定义顺序过滤和合并后，没有有效的数据集可供绘制。")
        print("请检查：")
        print("1. CSV 文件中的数据集名称是否与 `dataset_categories` 中的名称完全匹配（包括大小写和特殊字符）。")
        print("2. `my_operator.csv`, `framework1.csv`, `framework2.csv` 是否都包含您想比较的数据集。")
        exit()

    # 计算加速比
    merged_df['speedup_fw1'] = merged_df['time_fw1'] / merged_df['time_my']
    merged_df['speedup_fw2'] = merged_df['time_fw2'] / merged_df['time_my']

    # 准备绘图数据 (已按预定顺序排列)
    datasets = merged_df['dataset'].tolist()
    speedup1 = merged_df['speedup_fw1'].tolist()
    speedup2 = merged_df['speedup_fw2'].tolist()

    # --- 绘图 ---
    n_datasets = len(datasets)
    x = np.arange(n_datasets)  # x轴位置 [0, 1, ..., n_datasets-1]
    width = 0.35  # 柱状图的宽度

    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 0.8), 7)) # 增加图形高度以容纳分类标签

    # 绘制柱状图
    rects1 = ax.bar(x - width/2, np.minimum(speedup1, y_axis_limit), width, label=f'vs {framework1_name}', color=color_f1)
    rects2 = ax.bar(x + width/2, np.minimum(speedup2, y_axis_limit), width, label=f'vs {framework2_name}', color=color_f2)

    # 添加超过 y_axis_limit 的加速比数值标注
    def add_annotations(rects, speedups):
        for i, rect in enumerate(rects):
            if pd.notna(speedups[i]) and np.isfinite(speedups[i]) and speedups[i] > y_axis_limit:
                height = y_axis_limit
                ax.annotate(f'{speedups[i]:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, annotation_offset * y_axis_limit * 10),
                            textcoords="offset points", ha='center', va='bottom', fontsize=12)

    add_annotations(rects1, speedup1)
    add_annotations(rects2, speedup2)

    # --- 添加分类分隔线和标签 ---
    category_boundaries = []
    category_midpoints = []
    start_index = 0
    # 使用实际绘制的数据集来确定边界和中点
    current_datasets_in_plot = merged_df['dataset'].tolist()
    temp_datasets_by_cat = {cat: [] for cat in category_order}
    for ds in current_datasets_in_plot:
        for cat, ds_list in dataset_categories.items():
            if ds in ds_list:
                temp_datasets_by_cat[cat].append(ds)
                break

    current_index = 0
    for cat_name in category_order:
        datasets_in_this_cat = temp_datasets_by_cat[cat_name]
        num_datasets_in_cat = len(datasets_in_this_cat)

        if num_datasets_in_cat > 0:
            # 边界位于当前类别最后一个元素的右侧
            boundary = current_index + num_datasets_in_cat - 0.5
            category_boundaries.append(boundary)
            # 中点用于放置标签
            midpoint = current_index + (num_datasets_in_cat -1) / 2
            category_midpoints.append(midpoint)

            current_index += num_datasets_in_cat
        # else: 类别中没有数据集被绘制，跳过


    # 绘制垂直分隔线 (在类别之间)
    for boundary in category_boundaries[:-1]: # 不在最后一个类别后画线
        ax.axvline(boundary, color='grey', linestyle='--', linewidth=1, ymin=0, ymax=1.0)

    # 添加类别标签 (在X轴下方)
    # 使用 Axis 单位，这样标签位置不受 Y 轴数据范围影响
    # y 位置为负数表示在轴线的下方
    for i, cat_name in enumerate(category_order):
         # 仅当该类别有数据时才添加标签
        if i < len(category_midpoints):
            midpoint = category_midpoints[i]
            ax.text(midpoint, y_offset_for_type_labels, cat_name, ha='center', va='top', fontsize=10, fontweight='bold', transform=ax.get_xaxis_transform())
            # transform=ax.get_xaxis_transform() 使 x 按数据坐标，y 按轴坐标 (0=底部, 1=顶部)

    # --- 设置图表其余部分 ---
    ax.set_ylabel(y_axis_label)
    # ax.set_xlabel(x_axis_label) # X轴标签现在由下面的分类标签承担
    # ax.set_title(chart_title)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()

    # 设置Y轴上限
    ax.set_ylim(0, y_axis_limit * 1.1)

    # 添加水平网格线
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # 移除X轴刻度线下面的小标记 (tick marks)，因为我们有分类标签了
    ax.tick_params(axis='x', which='major', bottom=False)

    # 调整布局以防止标签重叠
    plt.subplots_adjust(bottom=0.25) # 增加底部边距以容纳分类标签和旋转的刻度标签
    # fig.tight_layout() # tight_layout 可能与 subplots_adjust 冲突，先尝试后者

    # 保存为 PDF
    plt.savefig(output_pdf_file, format='pdf', bbox_inches='tight')
    print(f"图表已成功保存为 '{output_pdf_file}'")

    # (可选) 显示图表
    # plt.show()

    # --- 计算总平均加速比 ---
    total_avg_speedup_fw1 = merged_df['speedup_fw1'].mean()
    total_avg_speedup_fw2 = merged_df['speedup_fw2'].mean()

    print(f"\n总平均加速比:")
    print(f"  相对于 {framework1_name}: {total_avg_speedup_fw1:.2f}x")
    print(f"  相对于 {framework2_name}: {total_avg_speedup_fw2:.2f}x")

    # --- 计算各个类别的平均加速比 ---
    print("\n各类别的平均加速比:")
    for category_name, datasets_in_category in dataset_categories.items():
        # 筛选属于当前类别的数据集
        category_df = merged_df[merged_df['dataset'].isin(datasets_in_category)]
        if not category_df.empty:
            avg_speedup_fw1 = category_df['speedup_fw1'].mean()
            avg_speedup_fw2 = category_df['speedup_fw2'].mean()
            print(f"  {category_name}:")
            print(f"    相对于 {framework1_name}: {avg_speedup_fw1:.2f}x")
            print(f"    相对于 {framework2_name}: {avg_speedup_fw2:.2f}x")
        else:
            print(f"  {category_name}: 无数据")

except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"发生未知错误: {e}")
    import traceback
    traceback.print_exc() # 打印详细错误信息