# 计划：Projection-based Cosine 量化（可执行版本）

## 1) 目标与成功标准

### 目标
在现有 ResidualQuantizer 基础上，引入 projection-space cosine 匹配，以降低 PA 数据集 Semantic ID 碰撞率。

### 成功标准
- 训练可稳定收敛（无 NaN、无大规模死码）
- 与当前 `use_cosine=True` 基线相比：碰撞率显著下降
- 阶段目标：PA 上碰撞率从约 8% 降到 `< 3%`

## 2) 关键设计（已修正）

### 设计 A：参数入口统一
- 不新增 `config['quantizer']` 顶层分组
- 统一放入 `config['model']`
- `SemanticIDModel -> ResidualQuantizer -> VectorQuantizer` 全链路显式透传

### 设计 B：检索空间与重构空间解耦
- 索引检索：在 projection 空间做 cosine（`argmax(sim)`）
- 量化输出：用原始码本向量做 scalar projection（不是投影后码本向量）

计算流程：
1. `z_proj = P(z), e_proj = P(E)` 用于相似度检索
2. `idx = argmax(cos(z_proj, e_proj))`
3. `e_raw = E[idx]`
4. `scalar = <z, e_raw> / (||e_raw||^2 + eps)`
5. `q = scalar * e_raw`

### 设计 C：初始化与训练稳定性
- K-means 初始化继续基于原始空间
- 增加 projection warmup（推荐前 N step 或 1 epoch 回退到原始匹配）
- EMA 仅维护 `embedding.weight`，`codebook_projection` 仅通过梯度更新

## 3) 逐文件 Patch 清单

### [A] `main/semantic_id/config.py`

1. 在 `DEFAULT_CONFIG['model']` 新增字段：

```python
'use_projection': True,
'projection_warmup_steps': 1000,
'projection_detach_codebook': False,
```

说明：
- `use_projection`：总开关
- `projection_warmup_steps`：warmup 阶段回退到原始匹配
- `projection_detach_codebook`：可选稳定项，默认 False（可做 ablation）

### [B] `main/semantic_id/model.py`

1. `SemanticIDModel.__init__()` 中实例化 `ResidualQuantizer` 时透传：

```python
use_cosine=kwargs.get('use_cosine', False),
use_ema=kwargs.get('use_ema', True),
use_projection=kwargs.get('use_projection', False),
projection_warmup_steps=kwargs.get('projection_warmup_steps', 0),
projection_detach_codebook=kwargs.get('projection_detach_codebook', False),
```

2. `RQVAE` 若仍被使用，同步透传同一组参数，避免双实现行为不一致。

### [C] `main/semantic_id/quantizer.py`

`C1.` `VectorQuantizer.__init__()` 新增参数：
- `use_projection: bool = False`
- `projection_warmup_steps: int = 0`
- `projection_detach_codebook: bool = False`

新增成员：

```python
self.use_projection = use_projection
self.projection_warmup_steps = projection_warmup_steps
self.projection_detach_codebook = projection_detach_codebook
self.register_buffer('_global_step', torch.zeros((), dtype=torch.long))

if self.use_projection:
    self.codebook_projection = nn.Linear(embedding_dim, embedding_dim)
    nn.init.normal_(self.codebook_projection.weight, std=embedding_dim ** -0.5)
    if self.codebook_projection.bias is not None:
        nn.init.zeros_(self.codebook_projection.bias)
```

`C2.` `VectorQuantizer.forward()` 在 `use_projection=True` 时替换匹配与量化逻辑：

```python
use_proj_now = (
    self.use_projection
    and self._initialized
    and (self._global_step.item() >= self.projection_warmup_steps)
)

if use_proj_now:
    z_match = self.codebook_projection(z)
    e_match = self.codebook_projection(self.embedding.weight)
    if self.projection_detach_codebook:
        e_match = e_match.detach()

    sim = torch.mm(F.normalize(z_match, dim=-1), F.normalize(e_match, dim=-1).t())
    indices = sim.argmax(dim=-1)

    # 关键：量化向量来自原始码本，而不是投影码本
    e_raw = F.embedding(indices, self.embedding.weight)
    dot_product = torch.sum(z * e_raw, dim=-1, keepdim=True)
    norm_sq = torch.sum(e_raw * e_raw, dim=-1, keepdim=True)
    quantized = dot_product / (norm_sq + 1e-8) * e_raw

    # cosine commitment
    commitment_loss = 1 - F.cosine_similarity(quantized.detach(), z, dim=-1)
    loss = self.commitment_cost * commitment_loss.mean()
else:
    # 保持当前分支（原始 cosine/欧氏 + 现有损失）
    ...

quantized = z + (quantized - z).detach()

if self.use_ema and self.training and self._initialized:
    self._ema_update(z, indices)

if self.training:
    self._global_step.add_(1)
```

`C3.` `VectorQuantizer._ema_update()`
- 保持 EMA 仅更新原始 `embedding.weight`
- 死码重置继续使用原始 `z` 样本替换 `embedding.weight`
- 不更新 `codebook_projection` 参数

`C4.` `ResidualQuantizer.__init__()`
- 新增并透传参数：`use_projection`, `projection_warmup_steps`, `projection_detach_codebook`
- 每层 `VectorQuantizer(...)` 传入上述参数
- 清理旧的 `self.use_projection=False` 层间投影逻辑（避免与新语义冲突）

### [D] `main/semantic_id/cosine_projection_plan.md`
- 保留为执行文档
- 每完成一个 patch 手动勾选

## 4) 最小改动实现顺序（按依赖）

1. 先改 config：新增 `model.use_projection` 等字段
2. 再改 model 透传：保证参数进入 quantizer
3. 再改 `VectorQuantizer`：实现 projection 匹配 + raw codebook scalar
4. 再改 `ResidualQuantizer`：补透传并清理旧 projection 逻辑
5. 最后最小验证：单 batch 前向、短训 1-2 epoch、再跑完整实验

## 5) 验证清单（可直接执行）

### 5.1 功能正确性
- [ ] `use_projection=False` 时，与当前行为一致
- [ ] `use_projection=True` 且 warmup 未结束时，可正常回退分支
- [ ] warmup 后切换到 projection 分支，loss 连续无明显突变
- [ ] `indices` 形状保持 `[B, num_quantizers]`

### 5.2 稳定性
- [ ] 训练前 1000 step 无 NaN
- [ ] 码本利用率无明显退化（active codes）
- [ ] 死码重置仍触发且可恢复利用率

### 5.3 效果
- [ ] 记录 baseline（当前主分支）PA 碰撞率
- [ ] 记录 projection 版本 PA 碰撞率
- [ ] 对比碰撞率、重构损失、活跃码字比例

## 6) 建议实验命令

```bash
python semantic_id/train.py --data_dir ./dataset/yelp/processed/PA \
  --preset base --epochs 100 --device cpu --run_name PA_proj_v1
```

建议额外做一组 ablation：
- `use_projection=False`（基线）
- `use_projection=True, projection_warmup_steps=0`
- `use_projection=True, projection_warmup_steps=1000`

## 7) 风险与回滚

### 风险
- projection 分支初期不稳定导致索引抖动
- cosine commitment 与原损失权重耦合，可能导致重构变差

### 回滚策略
- 一级回滚：`use_projection=False`
- 二级回滚：保持 `use_projection=True`，但增大 warmup steps
- 三级回滚：改回 MSE commitment（仅必要时）

## 8) 完成定义（DoD）

- 参数链路全通（config -> model -> residual -> vector）
- 训练与推理路径均可运行
- PA 结果有可复现实验记录（命令、配置、指标）
- 至少一组设置达到“碰撞率明显优于基线”
