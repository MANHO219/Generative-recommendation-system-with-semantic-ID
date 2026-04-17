# 第二章 问题定义和代表性方法 — 技术资料

> 整理时间：2026-04-15
> 来源：项目代码分析（model.py, quantizer.py, encoder.py, dataset.py, trainer.py, llm_finetune/*, inference/*, tool/semantic_id_analysis.py）

---

## 一、问题定义

### 1.1 POI 推荐问题定义

**用户-POI 交互序列**：
```
S_u = [(poi_1, t_1), (poi_2, t_2), ..., (poi_n, t_n)]
```
其中 $S_u$ 表示用户 $u$ 的历史交互序列，$poi_i$ 为第 $i$ 次交互的 POI，$t_i$ 为时间戳。

**Next-POI 预测任务**：
```
poi_{n+1} = argmax_{poi \in P} P(poi | S_u, context)
```
基于用户历史序列 $S_u$ 和上下文 context，预测下一个 POI。

### 1.2 语义 ID 的定义与表示

**Semantic ID 格式**：
```
SID(poi) = <a_p1><b_p2><c_p3>
例如：<a_12><b_34><c_56>
```

**冲突解决后缀格式**：
```
原始 SID: <a_12><b_34><c_56>
冲突后：<a_12><b_34><c_56><d_1>
        ↑ <d_N> = 第 N 个冲突 POI 的顺序索引
```

**设计说明**：早期曾考虑将 PlusCode 地理邻里编码注入 Prompt 以消歧，但可能引入噪声，后续简化为 `<d_xx>` 格式存储，与前三层码本保持一致的 `<X_YY>` 结构。

**格式说明**：
- `<a_X>` / `<b_X>` / `<c_X>`：分别对应 P1/P2/P3 三层量化索引
- `<d_N>`：冲突解决后缀，N 为 1-based 顺序索引

---

## 二、RQ-VAE 残差量化变分自编码器

### 2.1 模型架构

**基础架构：RQVAE 类**（model.py:25-200）

```
Input → Encoder → RQ → Decoder → Output
         ↓                        ↑
    [batch, input_dim]    [batch, input_dim]
              ↓
         [batch, embedding_dim]
              ↓
         残差量化 (RQ)
              ↓
         [batch, embedding_dim]
```

**核心参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| input_dim | 79 | 输入特征维度 |
| embedding_dim | 64 | 量化嵌入维度 |
| num_quantizers | 3 | 量化层数（产生 P1/P2/P3） |
| codebook_size | 128 | 每层码本大小 |
| commitment_cost | 0.25 | commitment loss 权重 |

### 2.2 Encoder 设计

**简化 Encoder**（model.py:252-261）：
```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, embedding_dim),  # 79 → 64
    nn.LayerNorm(embedding_dim),
    nn.GELU()
)
```

**原方案风格**（多层 MLP）：
```python
encoder_dims = [input_dim] + hidden_layers + [embedding_dim]
# [79, 512, 256, 128, 64]
```

### 2.3 残差量化器（ResidualQuantizer）

**核心机制**（quantizer.py:356-477）：

```python
# 伪代码
residual = z
quantized_sum = 0
for i in range(num_quantizers):
    quantized, indices_i, loss_i = quantizer_i(residual)  # 量化当前残差
    quantized_sum += quantized                            # 累加量化向量
    residual = residual - quantized                       # 计算新残差

indices = [batch_size, num_quantizers]  # P1, P2, P3 三层索引
```

**各层独立码本**：每层有独立的 VectorQuantizer，码本大小为 `codebook_size`

**EMA 更新机制**（quantizer.py:286-343）：
- 使用指数移动平均更新码本向量
- 死码重置：使用率低于平均值 10% 时，用随机样本重置

### 2.4 损失函数

```python
total_loss = (
    recon_weight * recon_loss +      # MSE(decoder_output, input)
    quant_weight * quant_loss +        # commitment_loss
    align_weight * align_loss +        # 1 - cosine_sim(quantized, encoded)
    diversity_weight * utilization_loss  # L1 码本均匀性约束
)
```

**关键设计**：
- `diversity_weight` 延迟启用（epoch >= 5），避免训练初期干扰重构学习
- `use_ema=True` 时，码本自动维持利用率，无需显式 diversity loss

---

## 三、多源特征编码器

### 3.1 特征构成

| 特征 | 维度 | 说明 |
|------|------|------|
| category_emb | 64 | 类别嵌入（SentenceTransformer 或可学习） |
| spatial_3d | 3 | 球坐标（lat/lon → 3D 单位球面） |
| fourier_time | 12 | Fourier 时间特征（6频率 × sin/cos） |
| attr_emb | 64 (可选) | Attributes 文本嵌入 |

**总维度**：79（无 attr_emb）或 143（启用 attr_emb）

### 3.2 各类编码器

**CategoryEncoder**（encoder.py:18-85）：
- 预训练嵌入（SentenceTransformer all-MiniLM-L6-v2）
- 可学习嵌入（fallback）
- 多标签平均池化

**Spatial 3D**（dataset.py:984-993）：
```python
lat_r = np.radians(lat)
lon_r = np.radians(lon)
x = np.cos(lat_r) * np.cos(lon_r)
y = np.cos(lat_r) * np.sin(lon_r)
z = np.sin(lat_r)
# 3D 单位球面坐标
```

**Fourier Time**（dataset.py:995-1005）：
```python
# 12维：6频率 × sin/cos
for k in range(1, 7):
    features.append(np.sin(2 * np.pi * k * mean_hour / 24))
    features.append(np.cos(2 * np.pi * k * mean_hour / 24))
```

**AttributeEncoder**（encoder.py:208-258）：
- 星级嵌入（11级）
- 数值特征编码（review_count, is_open）
- 融合层输出

### 3.3 PlusCodeEncoder（encoder.py:88-158）

**字符级编码**：
- 字符集：20 个有效字符（排除 AILO）
- 位置编码捕获空间层级信息

---

## 四、语义 ID 分布分析指标体系

### 4.1 已有指标

| 指标 | 说明 | 计算方法 |
|------|------|----------|
| 覆盖率 (Coverage) | 唯一 SID 数 / 总 POI 数 | n_prefixes / total_pois |
| 冲突率 (Collision Rate) | 发生 ID 冲突的 POI 比例 | 同 SID 的 POI 数 > 1 |
| PlusCode 使用率 | 需要附加后缀的 POI 比例 | disambig_count / total |
| 前缀分布熵 | 衡量 prefix 分布均匀程度 | $H = -\sum p_i \log(p_i)$ |
| Gini 系数 | 衡量分布不均匀程度 | $G = \frac{2\sum i c_i}{n\sum c_i} - \frac{n+1}{n}$ |

### 4.2 码本利用率分析

```python
def analyze_codebook_usage(self, indices):
    for i, usage in enumerate(get_codebook_usage(indices)):
        active_codes = (usage > 0).sum()
        entropy = -(usage * torch.log(usage + 1e-10)).sum()
        utilization = active_codes / codebook_size
```

---

## 五、QLoRA 微调配置

### 5.1 模型配置

**基础模型**：Qwen3-8B-Instruct

**量化配置**（llm_finetune/config.py）：
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)
```

### 5.2 LoRA 配置

```python
LoraConfig(
    r=16,              # 秩
    lora_alpha=32,    # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 5.3 训练超参数

| 参数 | 默认值 |
|------|--------|
| max_seq_length | 512 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 4 |
| learning_rate | 2e-4 |
| num_train_epochs | 3 |
| warmup_steps | 100 |

---

## 六、Trie 约束解码

### 6.1 TokenTrie 数据结构

```python
class TrieNode:
    children: Dict[int, TrieNode]
    is_end: bool

class TokenTrie:
    def add(self, token_ids): ...
    def traverse(self, token_ids) -> TrieNode: ...
    def next_tokens(self, token_ids) -> List[int]: ...
```

**核心操作**：
- `add`：将 Semantic ID 的 token 序列添加到 Trie
- `traverse`：沿已生成 token 遍历 Trie
- `next_tokens`：返回当前前缀允许的下一个 token 列表

### 6.2 TrieConstrainedLogitsProcessor

```python
class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # 对每个 batch：
        # 1. 获取已生成 token 序列
        # 2. 遍历 Trie 获取允许的 next tokens
        # 3. mask 掉不允许的 token
```

**约束逻辑**：
- 如果遍历 Trie 为 None（无效前缀）：mask 所有 token，只允许 EOS
- 如果遍历成功：只允许 `node.children` 中的 token
- 支持 early stop（当 `node.is_end=True` 时允许 EOS）

### 6.3 生成式评测指标

| 指标 | 说明 |
|------|------|
| exact_match | Top-1 与真实 SID 完全匹配的比例 |
| hit@K | 真实 SID 出现在 Top-K 的比例 |
| mrr@K | Mean Reciprocal Rank |
| valid_sid_rate | 生成的 SID 是合法 POI 的比例 |
| avg_latency_ms | 平均生成延迟（毫秒） |

---

## 七、Prompt 模板

### 7.1 系统 Prompt

```
You are an intelligent POI (Point of Interest) recommendation assistant.
Your task is to predict the next POI that a user will visit based on
their profile, visit history, and current spatiotemporal context.
Generate the Semantic ID in the format: <a_p1><b_p2><c_p3> (e.g., <a_12><b_34><c_56>).
```

### 7.2 用户输入模板

```
### User Profile:
- User ID: {user_id}
- Active Level: {review_count} reviews
- Average Rating: {average_stars:.1f} stars
- Favorite Categories: {favorite_categories}

### Spatiotemporal Context:
- Current Location: Plus Code {pluscode}
- Time: {time_description}
- Day Type: {day_type}

### Visit History (Recent {count} visits):
{history_items}

Based on the above information, predict the Semantic ID of the next POI
the user will visit.

Output only the Semantic ID in format: <a_XX><b_XX><c_XX> (e.g., <a_12><b_34><c_56>).
```

---

## 八、数据划分策略

> **说明**：本节描述的地理分层切分是针对**全量 Yelp 数据集**设计的。实际训练使用 Philadelphia 子集（单一城市），该切分策略意义有限，此处供参考。

### 8.1 地理分层切分（设计阶段，待验证）

```python
def geo_stratified_split(dataset, geo_map, train_ratio=0.8, val_ratio=0.1):
    # 按城市桶分组
    bucket_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        bucket = geo_map[bid].bucket
        bucket_to_indices[bucket].append(idx)

    # 每组内按比例切分
    for bucket, indices in bucket_to_indices.items():
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(indices[:n_train])
        ...
```

**作用**：确保每个城市的数据都按比例划分到 train/val/test，避免某城市数据只出现在测试集。

### 8.2 城市桶构建规则（设计阶段，待验证）

- POI 数 >= 200 的城市：独立桶（`state:city`）
- POI 数 < 200 的城市：合并到 `{state}:_other`

**实际训练**：直接使用 `dataset/yelp/processed/Philadelphia/` 子集数据，不经过地理分层切分。

---

## 九、RQ-VAE 三层量化流程

```
输入向量 z (64维)
  ↓
P1 量化: z → quantized_1, residual_1 = z - quantized_1
  ↓
P2 量化: residual_1 → quantized_2, residual_2 = residual_1 - quantized_2
  ↓
P3 量化: residual_2 → quantized_3
  ↓
最终量化向量: quantized_1 + quantized_2 + quantized_3
Semantic ID: [idx_P1, idx_P2, idx_P3]
```

---

*整理完成：2026-04-15*
