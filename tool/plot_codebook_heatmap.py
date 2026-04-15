"""
码本利用率热力图绘制脚本

用法：
    python tool/plot_codebook_heatmap.py

输出：
    analysis/PA_main_city/PA_main_city_layer1_heatmap.png
    analysis/PA_main_city/PA_main_city_layer2_heatmap.png
    analysis/PA_main_city/PA_main_city_layer3_heatmap.png
    analysis/PA_main_city/PA_main_city_codebook_heatmap_combined.png  (三合一拼接图)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

layer_names = ['PA_main_city_layer1_codes.csv', 'PA_main_city_layer2_codes.csv', 'PA_main_city_layer3_codes.csv']
grids = []

for layer_idx, fname in enumerate(layer_names, 1):
    df = pd.read_csv(f'analysis/PA_main_city/{fname}')
    usage = np.zeros(128)
    for _, row in df.iterrows():
        code = int(row['code'])
        usage[code] = row['n_pois']
    grids.append(usage.reshape(16, 8))

# 三合一拼接图
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

vmin = min(g.min() for g in grids)
vmax = max(g.max() for g in grids)

for idx, (ax, grid, layer) in enumerate(zip(axes, grids, [1, 2, 3])):
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    for i in range(16):
        for j in range(8):
            val = int(grid[i, j])
            if val > 0:
                text_color = 'white' if val > 150 else 'black'
                ax.text(j, i, val, ha='center', va='center', fontsize=6, color=text_color)
    ax.set_title(f'P{layer}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Code Group')
    ax.set_ylabel('Code Index')
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(8))
    ax.set_yticks(range(0, 16, 2))
    ax.set_yticklabels(range(0, 128, 16))

# 共享colorbar
fig.subplots_adjust(right=0.86, wspace=0.35)
cbar_ax = fig.add_axes([0.88, 0.15, 0.012, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Number of POIs')

plt.suptitle('Codebook Usage Heatmap - Philadelphia (11,711 POIs)', fontsize=14, y=1.02)
plt.savefig('analysis/PA_main_city/PA_main_city_codebook_heatmap_combined.png', dpi=150, bbox_inches='tight')
print('Saved: PA_main_city_codebook_heatmap_combined.png')
plt.close()
