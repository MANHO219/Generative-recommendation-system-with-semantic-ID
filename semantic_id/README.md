# Semantic ID 模块

基于 RQ-VAE（残差量化变分自编码器）的 POI 语义 ID 生成系统，支持地理分层数据切分与可选地理特征编码。

---

## 快速开始：训练

### 前置依赖

```bash
pip install torch einops pyyaml
# openlocationcode 可选，缺失时打印警告但不影响训练
```

### 训练命令

**GPU 训练（推荐）：**
```bash
cd D:\作业\毕业设计\main
python semantic_id/train.py \
    --data_dir ./dataset/yelp/processed \
    --preset base \
    --epochs 100 \
    --batch_size 128 \
    --device cuda
```

**CPU 训练（调试）：**
```bash
cd D:\作业\毕业设计\main
python semantic_id/train.py \
    --data_dir ./dataset/yelp/processed \
    --preset base \
    --epochs 20 \
    --batch_size 16 \
    --device cpu
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data/yelp/processed` | **必须手动指定**，实际路径为 `./dataset/yelp/processed` |
| `--preset` | `base` | 预设配置：`small` / `base` / `large` |
| `--device` | `cuda` | 训练设备，CUDA 不可用时自动回退到 CPU |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--resume` | - | 从检查点恢复训练（传入 `.pt` 路径） |
| `--config` | - | 自定义配置文件（JSON/YAML），覆盖默认值 |

### 预设配置对比

| 预设 | embedding_dim | num_quantizers | codebook_size | 适用场景 |
|------|---------------|----------------|---------------|--------|
| `small` | 128 | 2 | 32 | 快速验证 |
| `base` | 128 | 3 | 64 | 默认，平衡性能 |
| `large` | 512 | 4 | 128 | 追求更低冲突率 |

### 训练输出

```
./logs/semantic_id/          # 训练日志
./checkpoints/semantic_id/   # 模型检查点（best_model.pt, final_model.pt, codebooks.pt）
./output/semantic_ids.json   # 所有 POI 的 Semantic ID
```

---

## 项目结构

```
semantic_id/
├── train.py       # 训练入口脚本（自动选择数据集）
├── config.py      # 配置管理（DEFAULT_CONFIG + 预设模板 + GNPR-SID V2 配置）
├── dataset.py     # 数据加载（YelpPOIDataset / GNPRSIDPOIDataset）
├── encoder.py     # 特征编码器（MultiSourceEncoder、GeoEncoder）
├── quantizer.py   # 残差量化器（VectorQuantizer、ResidualQuantizer）
├── model.py       # 完整模型（RQVAE、SemanticIDModel）
├── decoder.py     # 解码器（SemanticDecoder）
├── trainer.py     # 训练器（SemanticIDTrainer）
└── __init__.py    # 模块导出
```

---

## 架构概述

### 方案 A：原方案（多源特征编码）

```
POI 特征
  ├─ 类别 (categories)         ┐
  ├─ 空间 (plus_code)          │  MultiSourceEncoder
  ├─ 时间 (checkin hours)      │  → 拼接 → MLP → embedding_dim
  ├─ 属性 (stars, is_open...)  │
  └─ 地理 (state/city, 可选)   ┘
         │
         ▼
   ResidualQuantizer
   ├─ VQ Layer 1 → code_1
   ├─ VQ Layer 2 → code_2  (量化残差)
   └─ VQ Layer 3 → code_3
         │
         ▼
  Semantic ID = (code_1, code_2, code_3)
```

### 方案 B：GNPR-SID V2（推荐）

```
POI 特征
  ├─ 类别语义 (SentenceTransformer) ─┐
  ├─ 空间 3D (球坐标)                │  直接拼接
  └─ 时间 Fourier (6频率 × 2)       ┘
         │
         ▼
   79 维特征向量
         │
         ▼
   Encoder: Linear(79→64) + LayerNorm + GELU
         │
         ▼
   ResidualQuantizer (3层, 每层64码)
         │
         ▼
  Semantic ID = (code_1-code_2-code_3)
```

**训练损失：**
- `recon_loss`：重构损失（MSE）
- `quant_loss`：VQ commitment loss
- 码本使用 EMA 更新 + 死码重置，自动维持利用率

---

## 地理分层改进（2025-03）

### 背景
原始 `random_split` 导致：
1. 码本学成"全局混合"，语义空间在跨州长尾场景下塌缩到高频码字
2. 用户序列未按城市过滤，序列信噪比低

### 实现方案

**阶段 A（已默认启用）：地理分层数据切分**
- `build_geo_partition()`：按 state/city 统计 POI 数，< 200 的小城合并至 `{state}:_other` 桶
- `geo_stratified_split()`：对每个城市桶内按比例切 train/val/test，确保每个城市均匀分布在三个 split
- 覆盖 117,788 个 POI，24 个州

**阶段 B（默认关闭）：地理特征 Embedding**
- `GeoEncoder`：state_emb + city_emb → Linear → LayerNorm → geo 向量
- 拼接到 MultiSourceEncoder 输出，提供显式地理先验
- 通过 `use_geo_features=True` 启用

**序列地理过滤**
- `YelpSequenceDataset` 支持 `geo_filter=True`：历史序列限制在同城，不足时回退到同州

### 配置开关

在 `config.py` 的 `geo` 节修改，或通过自定义配置文件覆盖：

```python
'geo': {
    'geo_stratified_split': True,   # 阶段A：地理分层切分（推荐保持开启）
    'min_city_poi_count': 200,      # 小城合并阈值
    'use_geo_features': False,      # 阶段B：地理特征编码（改变模型结构）
    'state_dim': 32,
    'city_dim': 64,
}
```

### 回滚方式

| 开关 | 值 | 效果 |
|------|-----|------|
| `geo_stratified_split` | `False` | 退回原 `random_split` |
| `use_geo_features` | `False`（默认）| 不实例化 GeoEncoder，模型结构不变 |
| `geo_filter` | `False` | 序列数据集退回原逻辑 |

---

## GNPR-SID V2 特征方案（推荐）

### 背景

原方案使用 Plus Code 字符编码 + 时段分布特征，特征维度较高（~256维）但信息冗余大，导致**码本严重坍缩**（117k POI → 14k 唯一 SID，碰撞率 88%）。

GNPR-SID V2 采用更紧凑的特征表示：
- **类别**: SentenceTransformer 语义嵌入（64维）
- **空间**: 3D 球坐标（lat/lon → 3维）
- **时间**: Fourier 时间特征（12维，捕捉周期性）
- **总计**: 79 维（vs 原方案 ~256 维）

### 启用方式

在 `config.py` 的 `gnpr_v2` 节设置：

```python
'gnpr_v2': {
    'use_gnpr_v2_features': True,           # 启用 GNPR-SID V2 特征
    'category_dim': 64,                    # 类别嵌入维度
    'spatial_dim': 3,                      # 球坐标维度（固定）
    'temporal_dim': 12,                    # Fourier 时间维度（固定）
    'use_sentence_transformer': False,     # 是否使用 SentenceTransformer
    'sentence_model_name': 'all-MiniLM-L6-v2',
    'category_embeddings_path': None,       # 或指定预训练嵌入路径
}
```

### 使用 SentenceTransformer（推荐）

```python
'gnpr_v2': {
    'use_gnpr_v2_features': True,
    'use_sentence_transformer': True,       # 首次运行会自动下载模型
    'sentence_model_name': 'all-MiniLM-L6-v2',
}
```

首次运行时会自动从 HuggingFace 下载模型（约 90MB），缓存到 `~/.cache/huggingface/hub/`。

### 架构变化

| 对比项 | 原方案 | GNPR-SID V2 |
|--------|--------|-------------|
| 特征维度 | ~256 | 79 |
| 类别编码 | 可学习嵌入 | SentenceTransformer（可选） |
| 空间编码 | Plus Code 字符级（10维） | 球坐标 3D（3维） |
| 时间编码 | 时段比例（6维） | Fourier 展开（12维） |
| 编码器 | MLP 多层 | 单层 Linear |
| 预期碰撞率 | 88% | < 10% |

### 训练命令

```bash
# GPU 训练
cd D:\作业\毕业设计\main
python semantic_id/train.py \
    --data_dir ./dataset/yelp/processed \
    --epochs 100 \
    --batch_size 128 \
    --device cuda

# CPU 训练（较慢）
python semantic_id/train.py \
    --data_dir ./dataset/yelp/processed \
    --epochs 20 \
    --batch_size 16 \
    --device cpu
```

训练完成后检查 `./output/semantic_ids.json` 的碰撞率，目标 < 10%。

---

## 评估指标

训练结束后自动输出：

| 指标 | 含义 | 目标 |
|------|------|------|
| `collision_rate` | 全局 SID 冲突率 | < 10%（GNPR-SID V2）/ < 5%（理想） |
| `avg_collision@city` | 同城内平均 SID 冲突率 | < 全局冲突率 |
| `max_collision@city` | 同城内最大 SID 冲突率 | 监控热点城市 |
| `codebook utilization` | 每层码本使用率 | > 60% |

---

## 已知问题与注意事项

1. **AMP 精度问题**：`use_amp=True`（GPU 训练默认启用）时，`quantizer.py` 的 EMA 更新需要显式 `.to(device=..., dtype=...)` 进行类型转换，否则 float16/float32 混用报错。已修复。

2. **`einops` 依赖**：`quantizer.py` 需要 `einops`，`requirements.txt` 已列出但未预装，需手动 `pip install einops`。

3. **`data_dir` 路径**：`train.py` 默认 `--data_dir` 为 `./data/yelp/processed`，实际数据在 `./dataset/yelp/processed`，**必须手动指定**。

4. **`num_workers`**：CPU 环境建议保持 `0`（主进程加载），Windows 多进程 DataLoader 存在兼容问题。

5. **码本坍缩问题**：原方案（多源特征编码）在 Yelp 数据上碰撞率高达 88%，导致 LLM 微调效果极差。**解决方案：启用 GNPR-SID V2 特征方案**，将碰撞率降至 < 10%。
