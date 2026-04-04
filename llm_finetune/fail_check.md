# LLM 微调失败分析

## 1. 推理日志分析

### 1.1 日志来源

文件：`D:\作业\毕业设计\main\logs\inference\final_log.md`

### 1.2 核心指标

| 指标 | 数值 |
|------|------|
| 总样本数 | 500 |
| 精确匹配 (exact_match) | 0 |
| 精确匹配率 | 0.00% |
| Hit@5 | 0.00% |
| Recall@5 | 0.00% |
| MRR@5 | 0.00% |
| 有效 SID 率 | 100% |
| 平均延迟 | 1899.82 ms |

### 1.3 预测结果模式

**关键发现：所有预测都是同一个值！**

```
top1=55-39-46<d_10> | gt=9-55-50<d_2>
top1=55-39-46<d_10> | gt=3-8-38<d_4>
top1=55-39-46<d_10> | gt=23-42-62<d_19>
...
top1=55-39-46<d_10> | gt=62-12-45<d_5>
```

**500 个样本，全部预测为 `55-39-46<d_10>`，无一例外！**

---

## 2. 问题诊断

### 2.1 模式崩塌 (Mode Collapse)

模型对所有输入都输出相同的 Semantic ID，这是典型的**模式崩塌**现象。

可能原因：

| 原因 | 描述 | 严重程度 |
|------|------|----------|
| Semantic ID 碰撞率过高 | 23.31% 的 POI 共享相同 SID | 🔴 严重 |
| 训练数据不足 | LLM 无法学习细粒度区分 | 🔴 严重 |
| Prompt 设计问题 | 模型无法理解任务 | 🟡 中等 |
| 温度/采样参数 | 生成过于保守 | 🟡 中等 |

### 2.2 碰撞率影响

当前 Semantic ID 碰撞率：**23.31%**

```
总 POI: 117,788
唯一 SID: 9,103
碰撞 POI: 108,685 (91.7%)
```

这意味着：
- `<a_55><b_39><c_46>` 这个 token 在 LLM 词汇中对应了 **大量不同的 POI**
- LLM 无法从这个 token 中区分"用户具体去过哪个 POI"
- 推荐任务退化为"猜测用户可能去过的某个大类"

### 2.3 碰撞案例

从日志可见，`55-39-46` 是高频碰撞 SID：

```
top1=55-39-46<d_10>  # d_10 表示这是第 11 个碰撞的 POI
```

这说明：
- `55-39-46` 这个基础 SID 有至少 11 个 POI 共享
- LLM 学到的 `<a_55><b_39><c_46>` 是这 11+ 个 POI 的**并集语义**
- 当用户访问其中某个 POI 时，模型只能预测到"这个大类"，无法精确定位

---

## 3. 根因分析

### 3.1 碰撞率 → 语义歧义 → LLM 无法学习

```
POI_A (collision) → <a_55><b_39><c_46><d_0>
POI_B (collision) → <a_55><b_39><c_46><d_1>
POI_C (collision) → <a_55><b_39><c_46><d_2>
...
POI_K (collision) → <a_55><b_39><c_46><d_10>
```

LLM 看到的训练数据：
- 用户去过 POI_A → 推荐 `<a_55><b_39><c_46><d_0>`
- 用户去过 POI_B → 推荐 `<a_55><b_39><c_46><d_1>`
- ...

**问题**：LLM 无法从历史序列中区分用户去过的是 A 还是 B，因为：
1. 历史序列中的 SID 可能是碰撞的
2. 推荐列表中的 SID 也是碰撞的
3. `<d_N>` 后缀在训练数据中可能没有被正确使用

### 3.2 训练数据质量问题

| 问题 | 影响 |
|------|------|
| 碰撞率 23.31% | 91.7% 的 POI 有碰撞 |
| 城市内碰撞率 1.34% | 同城内仍有碰撞 |
| 最大城市碰撞率 14.35% | 热点城市严重 |

---

## 4. 对比：原方案 vs GNPR-SID V2

### 4.1 方案对比

| 方案 | 唯一 SID | 碰撞率 | LLM 可用性 |
|------|----------|--------|------------|
| 原方案 | 9,103 | 23.31% | ❌ 失败 |
| GNPR-SID V2 (目标) | >100,000 | <10% | ✅ 预期可用 |

### 4.2 为什么 GNPR-SID V2 可能有效

GNPR-SID V2 的改进：
- **紧凑特征**：79 维 vs 256 维
- **连续空间编码**：球坐标 + Fourier
- **简化编码器**：单层 Linear，不易过拟合
- **预期碰撞率**：< 10%

如果碰撞率降至 10% 以下：
- 90%+ 的 POI 有唯一 SID
- LLM 可以学到"这个 POI 和那个 POI 是不同的"
- 推荐精度将大幅提升

---

## 5. 失败根因总结

### 直接原因

**LLM 模式崩塌**：所有输入都预测同一个 SID `55-39-46<d_10>`

### 根本原因

**Semantic ID 碰撞率过高 (23.31%)**

| 层级 | 问题 |
|------|------|
| 特征层 | 原始特征冗余 (256 维) |
| 编码层 | MLP 编码器过拟合 |
| 量化层 | 码本坍缩，64³ 容量浪费 |
| 应用层 | LLM 无法区分碰撞的 POI |

### 逻辑链条

```
原始特征 (256维, 冗余大)
    ↓
MLP Encoder (过拟合)
    ↓
RQ Quantizer (码本坍缩)
    ↓
碰撞率 23.31% (91.7% POI 碰撞)
    ↓
LLM 训练数据语义歧义
    ↓
LLM 模式崩塌 (所有预测 55-39-46<d_10>)
    ↓
Hit Rate = 0%
```

---

## 6. 解决方案

### 方案 A：继续使用原方案 (不推荐)

- 问题：碰撞率 23.31% 仍然太高
- 预期效果：仍会出现模式崩塌

### 方案 B：修复 GNPR-SID V2 并重新训练 (推荐)

1. 修复 Fourier/球坐标特征实现的 bug
2. 启用 SentenceTransformer 提升类别特征
3. 确保训练稳定完成 100+ epoch
4. 验证碰撞率 < 10%

### 方案 C：后处理解碰撞 (短期方案)

对碰撞的 POI，不使用 `<d_N>` 后缀，而是：
1. 在 Prompt 中加入 POI 的其他信息（名称、地址）
2. 让 LLM 额外学习从"类别+位置"推断具体 POI

### 方案 D：增大码本 (激进)

- 增加 `codebook_size` 从 64 到 128 或 256
- 增加 `num_quantizers` 从 3 到 4
- 码本容量：64³ = 262,144 → 128⁴ = 268,435,456

---

## 7. 下一步行动

1. **立即**：修复 GNPR-SID V2 实现 bug
2. **短期**：启用 SentenceTransformer，重新训练
3. **验证**：确认碰撞率 < 10% 后再进行 LLM 微调
4. **备选**：如果 GNPR-SID V2 无法修复，考虑方案 C/D

---

## 8. Prompt 设计对比分析

### 8.1 GNPR-SID V2 Prompt 格式

数据来源：`GNPR-SID-main/V1/datasets/NYC/train_id.json`

```json
{
  "instruction": "Here is a record of a user's POI accesses, your task is based on the history to predict the POI that the user is likely to access at the specified time.",
  "input": "User_2 visited: <2180> at 2012-04-20 07:23, <1796> at 2012-04-20 21:32, ... When 2012-05-01 17:35 user_2 is likely to visit:",
  "output": "<838>"
}
```

**格式化后** (使用 `format_fn`):

```
### Instruction:
Here is a record of a user's POI accesses, your task is based on the history to predict the POI that the user is likely to access at the specified time.

### Input:
User_2 visited: <2180> at 2012-04-20 07:23, <1796> at 2012-04-20 21:32, ... When 2012-05-01 17:35 user_2 is likely to visit:

### Response:
<838><|eot_id|>
```

### 8.2 你的 Prompt 设计

来源：`llm_finetune/config.py`

```python
PROMPT_TEMPLATE = {
    'system': (
        "You are an intelligent POI recommendation assistant. "
        "Your task is to predict the next POI that a user will visit based on "
        "their profile, visit history, and current spatiotemporal context. "
        "Generate the Semantic ID in the format: level0-level1-level2 (e.g., 12-34-56)."
    ),

    'user_template': (
        "### User Profile:\n"
        "- User ID: {user_id}\n"
        "- Active Level: {review_count} reviews\n"
        ...
    ),

    'context_template': (
        "### Spatiotemporal Context:\n"
        "- Current Location: Plus Code {pluscode}\n"
        "- Time: {time_description}\n"
        ...
    ),

    'history_template': (
        "### Visit History (Recent {count} visits):\n"
        "{history_items}"
    ),

    'instruction': (
        "Based on the above information, predict the Semantic ID of the next POI "
        "the user will visit. ...\n"
        "Output only the Semantic ID in format: XX-XX-XX"
    )
}
```

### 8.3 关键差异对比

| 维度 | GNPR-SID V2 | 你的设计 |
|------|-------------|----------|
| **格式** | 简单直接 | 分层复杂 |
| **SID 格式** | `<2180>` 整体作为 token | `21-35-42` 用连字符 |
| **Input 内容** | "visited: <SID> at <time>" | 拆成 profile/context/history 多部分 |
| **输出格式** | `<838>` 整体 | `XX-XX-XX` |
| **System prompt** | 无 (只用 Instruction) | 有 (详细描述任务) |
| **时间粒度** | 具体时间 2012-05-01 17:35 | 时间描述 "Tuesday evening" |
| **数据来源** | NYC 数据集 | Yelp 数据集 |

### 8.4 Prompt 设计问题分析

**问题 1：SID 分词不当**

GNPR-SID 用 `<2180>` 作为整体 token 嵌入，模型可以把整个 `<2180>` 当作一个语义单元学习。

你的 `21-35-42` 需要模型学习 3 个独立数字 token 的组合，复杂度更高。

**问题 2：Prompt 过于复杂**

你的设计把信息拆成 profile/context/history 多部分，可能导致：
- 模型过拟合于特定格式
- 学习噪声增加（冗余的用户信息）
- 关注点分散

**问题 3：缺少具体时间**

GNPR-SID：`When 2012-05-01 17:35 user_2 is likely to visit:`

你的设计：只说 "Time: Tuesday evening"，缺少具体时间点。

### 8.5 建议改进

采用 GNPR-SID 的简洁格式：

```python
PROMPT_TEMPLATE = {
    'instruction': "Here is a record of a user's POI visits. Based on the history, predict the next POI the user will visit at the specified time.",
    'input_template': "User_{user_id} visited: {history_items} At {target_time} the user is likely to visit:",
    'output_template': "<{semantic_id}>"
}
```

其中 `{history_items}` 格式：
```
<21-35-42> at 2024-01-01 12:00, <15-28-39> at 2024-01-02 18:30, <08-45-12> at 2024-01-03 20:15
```

**关键改进点**：

1. **SID 整体用 `<>` 包围**：让模型把 SID 当作单一语义单元
2. **去掉冗余的 profile 信息**：减少学习噪声
3. **使用具体时间**：帮助模型学习时空模式
4. **简化格式**：参考 GNPR-SID 的 "visited: <SID> at <time>" 格式

---

## 9. 日志文件

- Semantic ID 训练日志：`logs/semantic_id/train.log`
- LLM 推理日志：`logs/inference/final_log.md`
- 推理评估结果：`logs/inference/eval_final.json`
