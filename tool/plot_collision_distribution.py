"""
冲突规模分布柱状图绘制脚本

用法：
    python tool/plot_collision_distribution.py

输出：
    analysis/PA_main_city/PA_main_city_collision_distribution.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 设置中文字体（如果需要中文标签）
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

# 读取冲突数据
input_path = 'analysis/PA_main_city/PA_main_city_collision_tuples.csv'
df = pd.read_csv(input_path)

# 统计冲突规模分布
size_dist = Counter(df['n_pois'].values)

# 排序
sizes = sorted(size_dist.keys())
counts = [size_dist[s] for s in sizes]

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(range(len(sizes)), counts, color='#5daa5d', edgecolor='white', linewidth=0.5)

ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(sizes)
ax.set_xlabel('Collision Size (POIs per tuple)', fontsize=12)
ax.set_ylabel('Number of Tuples', fontsize=12)
ax.set_yscale('log')  # 对数刻度
ax.set_title('Collision Size Distribution - Philadelphia Dataset (11,711 POIs)', fontsize=13)

# 添加数值标签
for i, (s, c) in enumerate(zip(sizes, counts)):
    ax.text(i, c * 1.1, str(c), ha='center', va='bottom', fontsize=8, color='#333333')

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 保存
output_path = 'analysis/PA_main_city/PA_main_city_collision_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved: {output_path}')
