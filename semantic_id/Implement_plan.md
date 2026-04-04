Implement the following plan:

# 地理分层语义ID改进方案

## Context
训练效果差的根本原因：
1. `create_dataloaders()` 使用 `random_split`，完全无地理约束，码本学成"全局混合"
2. Yelp 跨州跨城长尾场景导致 SID 语义空间塌缩到高频码字
3. `YelpSequenceDataset` 用户序列未按城市过滤，序列信噪比低

**数据基础已确认：**
- `business_poi.json` 有 `city`/`state`/`plus_code_region`/`plus_code_city` 字段
- 117,788 个 POI，24 个州/省，Top-city: Philadelphia(14569), Tucson(9250), Tampa(9050)...
- `review_poi.json` 有 `date` 字段，支持时间切分

---

## 落地顺序（按阶段）

### 阶段 A：地理分层切分（只改数据切分，不改模型）

#### 改动1：`dataset.py` — 新增 `build_geo_partition()` 函数

位置：`create_dataloaders()` 之前新增全局函数

```python
def build_geo_partition(business_list, min_city_poi=200):
    """
    按 state/city 统计 POI 数，小城市合并至 {state}_other 桶
    返回:
      geo_map:    {business_id: {'state': str, 'city': str, 'bucket': str}}
      state_vocab:{state: idx}   # 0=PAD, 1=UNK, 2+为各州
      city_vocab: {bucket: idx}  # 0=PAD, 1=UNK, 2+为各桶
    """
```

逻辑：
- 统计每 `f"{state}:{city}"` 桶的 POI 数
- POI 数 < min_city_poi → 归入 `f"{state}:_other"` 桶
- 分别建 state_vocab（去重+排序）和 city_vocab

#### 改动2：`dataset.py` — 新增 `geo_stratified_split()` 函数

```python
def geo_stratified_split(dataset, geo_map, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    对每个城市桶内的索引按比例切 train/val/test
    确保每个桶内比例一致，避免某城市只出现在 test
    返回: (train_indices, val_indices, test_indices)
    """
```

逻辑：
- 按 `geo_map[biz_id]['bucket']` 将索引分组
- 每组内 shuffle 后按 [train_ratio, val_ratio, 1-train-val] 切分
- 合并各组的 train/val/test 后 shuffle

#### 改动3：`dataset.py` — `YelpPOIDataset` 加载 geo 字段

在 `__init__` 中：
- 从 `business_poi.json` 读 `city`/`state` 字段
- 调用 `build_geo_partition()` 建 `self.geo_map`、`self.state_vocab`、`self.city_vocab`
- 新增属性 `self.num_states`、`self.num_cities`

在 `__getitem__` 中，新增返回：
```python
'state_id': torch.tensor(self.state_vocab.get(state, 1))  # 1=UNK
'city_id':  torch.tensor(self.city_vocab.get(bucket, 1))
```

新增方法 `get_geo_vocab()` → 返回 `(state_vocab, city_vocab)`

#### 改动4：`dataset.py` — `create_dataloaders()` 替换 split 逻辑

新增参数：`geo_stratified=True`，`min_city_poi=200`

- 若 `geo_stratified=True`：用 `geo_stratified_split(dataset, dataset.geo_map, ...)` 替换 `random_split`
- 回传 `poi_dataset` 时同时附带 `geo_map` 供下游使用

#### 改动5：`dataset.py` — `poi_collate_fn()` 新增 geo 字段

```python
'state_id': torch.stack([item['state_id'] for item in batch])
'city_id':  torch.stack([item['city_id'] for item in batch])
```

---

### 阶段 B-1：地理特征 Embedding（config 开关，默认关闭）

#### 改动6：`encoder.py` — 新增 `GeoEncoder` 类

```python
class GeoEncoder(nn.Module):
    def __init__(self, num_states, num_cities, state_dim=32, city_dim=64):
        self.state_emb = nn.Embedding(num_states + 2, state_dim, padding_idx=0)
        self.city_emb  = nn.Embedding(num_cities + 2, city_dim, padding_idx=0)
        self.fusion    = nn.Linear(state_dim + city_dim, city_dim)
        self.norm      = nn.LayerNorm(city_dim)
    def forward(self, state_id, city_id):
        # cat → fusion → norm → [batch, city_dim]
```

#### 改动7：`encoder.py` — `MultiSourceEncoder` 可选拼接 geo

- `__init__` 新增参数 `use_geo_features=False`，`num_states=0`，`num_cities=0`
- 若启用：实例化 `GeoEncoder`，并在拼接维度计算中加 `city_dim`
- `forward(x, state_id=None, city_id=None)` — 若有 geo 则拼接后融合

#### 改动8：`model.py` — `SemanticIDModel.forward()` 传递 geo

- `forward(feature_vector, state_id=None, city_id=None)`
- 将 state_id/city_id 传入 encoder

#### 改动9：`config.py` — 新增 geo 配置节

```python
'geo': {
    'geo_stratified_split': True,   # 阶段A开关
    'min_city_poi_count': 200,      # 小城合并阈值
    'use_geo_features': False,      # 阶段B-1开关（默认关）
    'state_dim': 32,
    'city_dim': 64,
}
```

`get_model_config()` / `get_training_config()` 中透传 geo 参数。

---

### 用户序列地理切分（`YelpSequenceDataset`）

#### 改动10：`dataset.py` — `YelpSequenceDataset` 加 geo_filter

新增参数 `geo_filter=True`，`fallback_to_state=True`

在 `__init__` 中：
- 加载 `geo_map`（与 YelpPOIDataset 共用或重新构建）
- 按城市桶对 reviews 分组：`city_reviews[bucket] = [review...]`

在序列构建逻辑（`_build_sequences()`）中：
- 对每个 `(user, target_business, target_time)`：
  1. 获取 `target_city = geo_map[target_business_id]['bucket']`
  2. 过滤 user 历史：`同城 AND 时间 < target_time`
  3. 若同城历史 < min_seq_len：回退到同州历史（`fallback_to_state=True` 时）
  4. 若仍不足 min_seq_len：跳过该样本
  5. 截断到 max_len，按时间排序

时间切分：
- 在同城内按时间切 train/val/test，避免数据泄漏
- 可复用 `geo_stratified_split` 时间变体

---

### 评估升级（`trainer.py`）

#### 改动11：`evaluate_semantic_ids()` 增加 geo-aware 指标

需要 `dataset.geo_map` 作为额外参数：

```python
def evaluate_semantic_ids(self, data_loader, geo_map=None):
    ...
    if geo_map:
        # collision_rate@city: 同城内 SID 冲突率
        # geo_consistency: SID 相同前缀的 POI 是否来自同城
        city_groups = defaultdict(list)  # bucket -> [sid]
        for biz_id, sid in semantic_ids.items():
            bucket = geo_map[biz_id]['bucket']
            city_groups[bucket].append(sid)
        per_city_collision = {
            b: 1 - len(set(sids)) / len(sids)
            for b, sids in city_groups.items()
        }
        metrics['avg_collision@city'] = np.mean(list(per_city_collision.values()))
        metrics['max_collision@city'] = np.max(list(per_city_collision.values()))
```

---

## 关键文件清单

| 文件 | 改动性质 | 核心改动点 |
|------|---------|-----------|
| `dataset.py` | 主要 | 新增函数×2，修改类×2，修改函数×2 |
| `encoder.py` | 中等 | 新增 `GeoEncoder`，修改 `MultiSourceEncoder` |
| `model.py` | 小 | `forward()` 透传 geo 参数 |
| `config.py` | 小 | 新增 `geo` 配置节 |
| `trainer.py` | 小 | `evaluate_semantic_ids()` 增加 geo 指标 |
| `train.py` | 极小 | 传递 `geo_map` 到 evaluate 调用 |

---

## 验证方式

1. **运行分层后的数据检查**：打印每个 split 中 state 分布，确认无极端不均衡
2. **训练跑通**：`python train.py --data_dir ./dataset/yelp/processed`
3. **评估指标对比**：
   - `global collision_rate` 应维持低位（< 5%）
   - `collision_rate@city` < `global collision_rate`（同城更少冲突）
   - `codebook utilization` > 60%

---

## 回滚点

- 阶段A：`create_dataloaders()` 保留 `geo_stratified=False` 分支，可秒切回原 `random_split`
- 阶段B：`use_geo_features=False`（默认）时，encoder 不实例化 `GeoEncoder`，模型结构与原来完全一致
- `YelpSequenceDataset` 的 `geo_filter=False` 退回原逻辑