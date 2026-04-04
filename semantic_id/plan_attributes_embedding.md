# 为 GNPR-SID V2 添加 Attributes 文本语义嵌入

## Context

当前 GNPR-SID V2 使用 79 维特征（category_emb 64D + spatial_3d 3D + fourier_time 12D），Philadelphia 训练后冲突率 22%。business_poi.json 中的 `attributes` 字段（价格等级、服务方式、氛围等）尚未利用。

原始方案用 one-hot / binary 编码 attributes，效果有限，原因：
- one-hot 丢失 ordinal 关系（NoiseLevel quiet/average/loud）
- 嵌套 dict 聚合太 lossy
- 和 category embedding 语义空间不同，拼接后 64D bottleneck 难以平衡

**新方案**：把 attributes 转成文本描述，用同一个 SentenceTransformer 编码为 64D 向量，与 category embedding 拼接。复用同一语义空间，自然保留 ordinal 和复杂语义。

## 特征维度变化

| 特征 | 维度 | 编码方式 |
|------|------|---------|
| category_emb | 64 | SentenceTransformer |
| spatial_3d | 3 | 球坐标 (lat/lon → 3D) |
| fourier_time | 12 | Fourier 时间 (6 freq × sin/cos) |
| **attr_emb** | **64** | **SentenceTransformer attributes 文本** |
| **总计** | **143** | |

## Attributes 文本构造方法

从 business_poi.json 的 attributes 字段提取关键属性，拼接为自然语言描述：

```python
def attributes_to_text(biz: Dict) -> str:
    """
    将 business attributes 转成文本描述。
    输入示例:
      {'RestaurantsPriceRange2': '2',
       'WiFi': "u'free'",
       'Alcohol': "u'none'",
       'NoiseLevel': "u'average'",
       'OutdoorSeating': 'True',
       'RestaurantsDelivery': 'False',
       'Ambience': "{'touristy': False, 'casual': True, ...}",
       'GoodForMeal': "{'breakfast': True, 'lunch': True, ...}"}
    """
    attrs = biz.get('attributes') or {}
    parts = []

    # 价格等级
    price = attrs.get('RestaurantsPriceRange2', '')
    if price:
        parts.append(f"Price range {price}")

    # WiFi
    wifi = _clean_str(attrs.get('WiFi', ''))
    if wifi and wifi not in ('none', 'no'):
        parts.append(f"WiFi {wifi}")

    # Alcohol
    alcohol = _clean_str(attrs.get('Alcohol', ''))
    if alcohol and alcohol not in ('none', 'no'):
        parts.append(f"Alcohol {alcohol}")

    # NoiseLevel
    noise = _clean_str(attrs.get('NoiseLevel', ''))
    if noise:
        parts.append(f"Noise level {noise}")

    # 简单 True/False 属性（只取有意义的）
    for attr, key in [
        ('OutdoorSeating', 'Outdoor seating'),
        ('RestaurantsTakeOut', 'Takeout available'),
        ('RestaurantsDelivery', 'Delivery available'),
        ('HasTV', 'Has TV'),
        ('GoodForKids', 'Good for kids'),
        ('RestaurantsReservations', 'Reservations accepted'),
        ('Caters', 'Caters'),
        ('BusinessAcceptsCreditCards', 'Credit cards accepted'),
        ('HappyHour', 'Happy hour'),
    ]:
        val = attrs.get(attr, '')
        if _is_true(val):
            parts.append(key)

    # 嵌套 dict：提取 True 的子键
    for nested_key in ['Ambience', 'GoodForMeal', 'BusinessParking', 'Music']:
        nested = attrs.get(nested_key, '')
        true_keys = _extract_true_keys(nested)
        if true_keys:
            parts.append(f"{nested_key}: {', '.join(true_keys)}")

    text = ', '.join(parts)
    return text if text else "No specific attributes"
```

### 辅助解析函数

```python
def _clean_str(val) -> str:
    """清洗 u'free', 'free', "True" 等混乱格式"""
    if val is None:
        return ''
    s = str(val).strip()
    s = s.lstrip("u").strip("'\"")
    return s.lower()

def _is_true(val) -> bool:
    """判断是否为 True"""
    if val is None:
        return False
    s = _clean_str(val)
    return s in ('true', '1', 'yes')

def _extract_true_keys(nested_str: str) -> List[str]:
    """从嵌套 dict 字符串提取值为 True 的键"""
    if not nested_str or nested_str.lower() == 'none':
        return []
    try:
        import ast
        d = ast.literal_eval(nested_str)
        if isinstance(d, dict):
            return [k for k, v in d.items() if v is True]
    except:
        pass
    return []
```

### 输出文本示例

```
Input:  {'RestaurantsPriceRange2': '2', 'WiFi': "u'free'", 'OutdoorSeating': 'True',
         'Ambience': "{'casual': True, 'romantic': True, 'intimate': False}"}

Output: "Price range 2, WiFi free, Outdoor seating, Ambience: casual, romantic"
```

## 修改文件清单

### 1. `d:/作业/毕业设计/main/semantic_id/dataset.py`

**修改 `GNPRSIDPOIDataset.__init__`**（约 line 848-882）：
- 调用 `self._load_attributes_text()`
- 调用 `self._load_attributes_embeddings(sentence_model_name)`

**添加新方法**：
```python
def _load_attributes_text(self) -> List[str]: ...
def _load_attributes_embeddings(self, model_name: str): ...
def _attributes_to_text(self, biz: Dict) -> str: ...
def _clean_str(self, val) -> str: ...
def _is_true(self, val) -> bool: ...
def _extract_true_keys(self, nested_str: str) -> List[str]: ...
```

**修改 `__getitem__`**（约 line 1027-1040）：
```python
attr_emb = self.attr_embeddings[idx]  # [64] 新增
feature_vector = np.concatenate([cat_emb, spatial, temporal, attr_emb])  # [143]
```

**修改 `feature_dim`**（约 line 869）：
```python
self.attribute_dim = 64
self.feature_dim = self.category_dim + 3 + 12 + self.attribute_dim  # 143
```

### 2. `d:/作业/毕业设计/main/semantic_id/config.py`

```python
'gnpr_v2': {
    'use_gnpr_v2_features': True,
    'category_dim': 64,
    'spatial_dim': 3,
    'temporal_dim': 12,
    'attribute_dim': 64,                  # 新增
    'use_attributes': True,               # 新增
    'use_sentence_transformer': True,
    'sentence_model_name': 'all-MiniLM-L6-v2',
    'category_embeddings_path': None,
}
```

### 3. `d:/作业/毕业设计/main/semantic_id/model.py`

修改 `create_model()` 中 input_dim 计算（约 line 576）：
```python
input_dim = (
    gnpr_v2_config.get('category_dim', 64) +
    gnpr_v2_config.get('spatial_dim', 3) +
    gnpr_v2_config.get('temporal_dim', 12) +
    gnpr_v2_config.get('attribute_dim', 0)  # 新增，默认 64
)
# 默认 64+3+12+64 = 143
```

### 4. `d:/作业/毕业设计/main/semantic_id/train.py`

**无需修改**。dataset 内部处理 attributes 编码，train.py 透传 config 即可。

## 验证步骤

1. **语法检查**：
   ```bash
   python -m py_compile semantic_id/dataset.py
   python -m py_compile semantic_id/model.py
   python -m py_compile semantic_id/config.py
   ```

2. **加载验证（POI 数量和特征维度）**：
   ```python
   from semantic_id.dataset import GNPRSIDPOIDataset
   ds = GNPRSIDPOIDataset('./dataset/yelp/processed/PA/Philadelphia',
                           use_sentence_transformer=True)
   print(ds.feature_dim)           # 应输出 143
   print(ds.attr_embeddings.shape) # 应输出 (11711, 64)
   item = ds[0]
   print(item['feature_vector'].shape)  # torch.Size([143])
   print(ds.attributes_text[0])   # 查看 attributes 文本内容
   ```

3. **训练验证（1 epoch）**：
   ```bash
   python semantic_id/train.py \
       --data_dir ./dataset/yelp/processed/PA/Philadelphia \
       --preset base --epochs 1 --batch_size 32 \
       --device cpu --run_name PA_philly_attrs_emb
   ```

4. **完整训练 + 冲突率对比**：
   ```bash
   python tool/sid_statistics.py --experiment PA_philly_attrs_emb

   # 对比
   # PA_main_city（无 attributes）: 22.0%
   # PA_philly_attrs_emb（有 attributes）: 预期下降
   ```

## 注意事项

- attributes 文本生成在 `__init__` 时一次性完成，不需要每次 `__getitem__` 重复解析
- SentenceTransformer 对空文本会有输出（编码器不会报错），`"No specific attributes"` 作为默认值
- 非餐饮 POI（Shopping、Medical 等）的 attributes 字段多为空，文本很短但仍可编码，不影响训练

## 关键细节补充（建议纳入实施）

1. **缓存策略（避免重复编码）**
   - 在 `dataset.py` 中为 `attr_embeddings` 增加磁盘缓存（如 `*.npy` + `*.json` metadata）。
   - cache key 建议包含：`data_dir`、`sentence_model_name`、POI 数量、attributes 文本哈希。
   - 命中缓存时直接加载；仅在 key 变化时重算并覆盖缓存。

2. **维度来源统一（避免 64 硬编码漂移）**
   - `self.attribute_dim` 从真实编码结果读取：`self.attr_embeddings.shape[1]`。
   - `self.feature_dim` 统一基于 `category_dim + spatial_dim + temporal_dim + attribute_dim` 动态计算。
   - 在 `model.py`/dataset 对接处添加断言，确保 `feature_vector.shape[-1] == input_dim`。

3. **缺失 attributes 文本策略（减少同质化）**
   - 将默认文本从单一 `"No specific attributes"` 扩展为可区分模板：
     - truly missing attributes
     - non-food category with sparse attributes
     - unknown attributes format
   - 目标是减少大批空 attributes POI 的向量塌缩到同一点。

4. **资源与离线回退策略（提升可用性）**
   - SentenceTransformer 加载失败、无网络或模型不可用时，回退到可用方案（如全零向量或旧版 one-hot/binary）。
   - 在日志中明确记录当前使用的是 `st_embedding` 还是 `fallback`，保证实验可解释。

5. **指标定义补全（避免单指标优化）**
   - 除 collision rate 外，固定跟踪：
     - codebook utilization
     - assignment entropy
     - retrieval recall@k（或任务相关下游指标）
   - 验收标准建议写成：`collision 降低` 且 `语义检索指标不下降`。
