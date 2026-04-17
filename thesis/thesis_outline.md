# 毕业论文开题资料整理

> 生成时间：2026-04-10
> 访谈模式：Deep Interview（Socratic Q&A）

---

## 一、项目背景

### 1.1 项目概述

本项目是一个基于 **Semantic ID（语义ID）** 的 POI（兴趣点）推荐系统，主要包含三个模块：

| 模块 | 技术方案 | 说明 |
|------|---------|------|
| **语义ID生成** | RQ-VAE（残差量化变分自编码器） | 将POI编码为三层语义ID（格式：`<a_P1><b_P2><c_P3>`） |
| **序列推荐** | QLoRA 微调 Qwen3-8B-Instruct | 基于用户历史行为预测下一个POI |
| **推理解码** | Trie 约束解码 | 确保生成的语义ID合法有效 |

### 1.2 项目结构

```
main/
├── semantic_id/          # RQ-VAE 语义ID生成
│   ├── model.py           # RQVAE 架构
│   ├── trainer.py         # 训练循环
│   └── dataset.py         # 数据集处理
├── llm_finetune/          # QLoRA 微调
│   ├── trainer.py         # SFTTrainer 微调
│   ├── config.py          # 配置（Prompt模板、LoRA参数）
│   └── dataset.py         # 微调数据构建
├── inference/             # 推理
│   ├── run_inference.py   # 主推理脚本
│   ├── constrained_decoding.py  # TrieConstrainedLogitsProcessor
│   └── trie.py            # TokenTrie 实现
├── tool/                  # 工具脚本
│   └── semantic_id_analysis.py  # 语义ID分布分析
├── dataset/yelp/          # Yelp数据集
│   └── processed/Philadelphia/   # Philadelphia子集
├── checkpoints/           # 模型权重
│   ├── semantic_id/       # RQ-VAE checkpoints
│   └── llm_sid/          # LoRA adapters
└── output/                # 输出
    └── sid/PA_main_city/ # 语义ID输出
```

### 1.3 技术栈

- **环境**：Conda 环境 `rqvae_cpu`（RQ-VAE训练）、`pytorch_env`（LLM微调）
- **基础模型**：Qwen3-8B-Instruct（Local）
- **微调方法**：QLoRA（4bit NF4量化，LoRA r=16, alpha=32）
- **数据集**：Yelp（Philadelphia子集， postal_code / plus_code_neighborhood）

---

## 二、核心创新点

### 2.1 创新点总结

| 创新点 | 类型 | 说明 |
|--------|------|------|
| **PlusCode 冲突解决** | 工程改进 | 当Semantic ID冲突时，附加PlusCode后缀区分 |
| **语义ID分布分析** | 方法论贡献 | 建立覆盖率、冲突率、熵、集中度等评估指标体系 |
| **端到端Pipeline** | 系统集成 | RQ-VAE + QLoRA + Trie约束解码完整流程 |

### 2.2 创新点定位策略

**叙事策略（推荐）**：

- **PlusCode 不作为核心创新宣传**，而是作为"证明方案可行"的工程实现
- 放在第三章（3.2节）作为系统设计的一部分，重点讲如何集成
- 在消融实验（5.4节）中验证其作用

- **语义ID分布分析作为核心方法论贡献**
- 放在第五章（5.2节）单独分析，建立评估框架
- 这是真正原创的评估指标体系

### 2.3 数据格式

**Semantic ID 格式（GNPR 风格）**：
```
<a_P1><b_P2><c_P3>
例如：<a_12><b_34><c_56>
```

**冲突解决后缀格式（GNPR disambig）**：
```
原始SID: <a_12><b_34><c_56>
冲突后：<a_12><b_34><c_56><d_1>
        ↑ <d_N> 顺序索引后缀，N 表示该 SID 对应的第 N 个 POI
```

**格式说明**：
- `<a_X>` / `<b_X>` / `<c_X>`：分别对应 P1/P2/P3 三层量化索引
- `<d_N>`：冲突解决后缀，N 为 1-based 顺序索引（当多个 POI 共享相同 SID 时区分）
- 与 GNPR v2 保持一致，采用角括号风格而非传统的 `P1-P2-P3[GRID]` 格式

---

## 三、Semantic ID 分布分析指标体系

### 3.1 已有指标（tool/semantic_id_analysis.py）

| 指标 | 现状 | 说明 |
|------|------|------|
| 覆盖率 (Coverage) | ✅ 已有 | 成功分配语义ID的POI比例 |
| 冲突率 (Collision Rate) | ✅ 已有 | 发生ID冲突的POI比例（未加PlusCode前） |
| PlusCode 使用率 | ✅ 已有 | 最终需要附加PlusCode后缀的POI比例 |
| Prefix分布 (Category/Region) | ✅ 已有 | 按prefix分组统计Top-K POI类别和地区 |

### 3.2 推荐补充指标

| 指标 | 用途 | 计算难度 |
|------|------|----------|
| **信息熵 (Entropy)** | 衡量prefix分布均匀程度 | 简单 |
| **Top-K 集中度** | 衡量是否有热点prefix | 简单 |
| **Gini 系数** | 衡量分布不均匀程度 | 简单 |

### 3.3 指标计算公式

```python
# 覆盖率
Coverage = covered_pois / total_pois × 100%

# 冲突率
Collision_Rate = conflicted_pois / covered_pois × 100%

# PlusCode 使用率
PlusCode_Rate = pluscode_pois / total_pois × 100%

# 信息熵
H = -Σ p_i × log(p_i)  # 均匀分布时熵最大

# Top-K 集中度
TopK_Concentration = Σ_{i=1}^{K} count(prefix_i) / total_pois
```

---

## 四、推荐评估指标

### 4.1 标准 POI 推荐指标

| 指标 | 说明 | 适用场景 |
|------|------|---------|
| **Recall@K** | 推荐的K个POI中包含真实POI的比例 | 最核心指标 |
| **NDCG@K** | 考虑排序质量的归一化折现增益 | 衡量排序合理性 |
| **MRR** | 第一个正确推荐出现位置的倒数均值 | 衡量命中速度 |
| **Hit Rate@K** | K次推荐中至少命中一次的用户的比例 | 用户满意度视角 |

### 4.2 Baseline 对比方法

- **FPMC**: Rendle et al., 2010, RecSys — "Factorizing Personalized Markov Chains for Next-Basket Recommendation"（引用论文已有 Yelp 数据集结果）
- **STAN**: Wang et al., 2021, KDD — "Spatio-Temporal Attention Network for Next POI Recommendation"（引用论文已有 Yelp 数据集结果）
- **SARS**: Liu et al., 2020, RecSys — "SARS: Sequential Aware Recommender System"（引用论文已有 Yelp 数据集结果）
- **GNPR**: Yang et al., 2020, WWW — "Geographic POI Recommender"（引用论文已有 Yelp 数据集结果）

**传统POI推荐方法**：FPMC, STAN, etc.
**基于ID的推荐方法**：SARS (RecSys 2020), GNPR, etc.

### 4.3 消融实验设计

| 实验 | 说明 |
|------|------|
| PlusCode 冲突解决的作用 | 无PlusCode vs 有PlusCode 的效果对比 |
| Trie 约束解码的作用 | 无约束 vs 有约束的解码成功率对比 |

**统计显著性检验协议**：
- **配对 t-test (paired t-test)**：对同一组用户在有/无某模块条件下进行配对比较，适用于评估 PlusCode 和 Trie 约束的个体影响
- **PlusCode 有/无显著性检验**：对使用 PlusCode 前后的 Recall@K / NDCG@K 指标差值进行配对 t-test，验证冲突解决模块的统计显著性
- **Trie 约束有/无显著性检验**：对有约束解码与无约束解码的生成成功率进行配对 t-test，验证约束解码的统计显著性
- **p-value 阈值**：p < 0.05 视为统计显著

---

## 五、论文大纲

### 第一章 绪论

- 1.1 选题背景与意义
  - POI推荐的研究背景
  - Semantic ID在推荐系统中的应用价值
  - 本研究的意义
- 1.2 国内外研究现状和相关工作
  - 传统POI推荐方法（FPMC, STAN, etc.）
  - 基于ID的序列化推荐（SRS, GNPR, etc.）
  - 基于语义嵌入的推荐方法
  - LLM在推荐系统中的应用
- 1.3 本文主要研究内容
  - 系统架构概述
  - PlusCode冲突解决机制
  - 语义ID分布分析
  - 实验设计与验证

### 第二章 问题定义和代表性方法

- 2.1 POI推荐问题定义
  - 用户-POI交互序列建模
    ```
    S_u = [(poi_1, t_1), (poi_2, t_2), ..., (poi_n, t_n)]
    其中 S_u 表示用户 u 的历史交互序列，poi_i 为第 i 次交互的POI，t_i 为时间戳
    ```
  - Next-POI预测任务定义
    ```
    poi_{n+1} = argmax_{poi \in P} P(poi | S_u, context)
    基于用户历史序列 S_u 和上下文 context，预测下一个POI
    ```
- 语义ID的定义与表示
    ```
    SID(poi) = <a_p1><b_p2><c_p3>
    其中 <a_p1>、<b_p2>、<c_p3> 为三层量化索引，冲突时附加 <d_n> 后缀
    ```
  - RQ-VAE残差量化变分自编码器
  - GNPR-SID v2 架构解析
  - QLoRA大模型微调方法
  - Trie约束解码

### 第三章 基于RQ-VAE的语义ID生成与冲突解决

- 3.1 多源特征编码器
  - 特征构成（类别、空间、时间、Fourier）
  - 类别编码器、空间坐标编码、Fourier时间特征
- 3.2 RQ-VAE语义ID生成框架
  - 编码器设计
  - 残差量化器（Quantizer）
  - 三层语义表示（P1/P2/P3）
- 3.3 三层量化流程
- 3.4 损失函数与训练策略
  - 总损失函数、重构损失、对齐损失、多样性损失
  - 动态多样性权重Warm-up策略
- 3.5 冲突解决机制
  - 冲突检测与判断
  - 顺序索引后缀附加策略
  - 冲突解决算法描述
- 3.6 语义ID分布质量分析
  - 覆盖率与冲突率
  - 分布熵与集中度
  - 码本利用率分析、死码重置
- 3.7 本章小结

### 第四章 基于QLoRA-LLM的POI序列推荐

- 4.1 指令微调数据构建
  - SID+时间历史Prompt模板
  - 训练数据生成流程
- 4.2 QLoRA微调策略
  - 量化配置（4bit NF4）
  - LoRA配置与训练超参数
  - 序列长度与Batch Size设置
- 4.3 Trie约束解码
  - TokenTrie数据结构
  - TrieConstrainedLogitsProcessor
  - 基于语义ID前缀的约束
- 4.4 实验设置
  - 数据集：Yelp-Philadelphia
  - 数据预处理：train/val/test split（80%/10%/10%）
  - Baseline方法（传统方法 + ID-based方法）
  - 评估指标：HR@K, MRR@K, NDCG@K
- 4.5 推荐效果对比实验
  - 与Baseline方法的HR@K / MRR@K / NDCG@K对比
  - 不同K值的性能分析
- 4.6 消融实验
  - 冲突解决机制的作用
  - Trie约束解码的作用
- 4.7 本章小结

### 第五章 总结与展望

- 6.1 研究总结
- 6.2 工作展望

### 章节 Limitations（补充在第六章前）

- 7.1 地理覆盖范围局限
  - Philadelphia 子集仅覆盖单一城市，模型在其他城市的泛化能力未验证
- 7.2 数据稀疏性问题
  - Yelp 用户行为数据的采样偏差对推荐效果的影响
- 7.3 语义ID层级设计局限
  - 三层Quantizer的codebook大小为超参数，需针对不同数据集调优
- 7.4 计算成本
  - RQ-VAE 训练和 Trie 约束解码的计算开销随 POI 规模线性增长

---

## 六、后续工作

### 6.1 近期待完成

- [ ] 补充信息熵、Gini系数等分布指标计算
- [ ] 实现 Recall@K, NDCG@K, MRR 评估脚本
- [ ] 完成 Philadelphia 城市的数据实验
- [ ] 补充 Baseline 对比实验

### 6.2 扩展方向

- [ ] 扩展到其他大城市（CA, TX等）
- [ ] 对比更多 Baseline 方法
- [ ] 分析不同 Prefix 长度（P1/P2/P3）对推荐效果的影响

---

## 七、Prompt 模板参考

### 系统 Prompt

```
You are an intelligent POI (Point of Interest) recommendation assistant.
Your task is to predict the next POI that a user will visit based on
their profile, visit history, and current spatiotemporal context.
Generate the Semantic ID in the format: <a_p1><b_p2><c_p3> (e.g., <a_12><b_34><c_56>).
```

### 用户输入模板

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
the user will visit. Consider:
1. User's historical preferences
2. Spatiotemporal patterns (location, time, day)
3. POI availability (must be open at the predicted time)

Output only the Semantic ID in format: <a_XX><b_XX><c_XX> (e.g., <a_12><b_34><c_56>).
```

---

## 八、关键文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| RQ-VAE 模型 | `semantic_id/model.py` | RQVAE架构 |
| RQ-VAE 训练 | `semantic_id/train.py` | 训练入口 |
| 语义ID分析 | `tool/semantic_id_analysis.py` | 分布分析工具 |
| LLM 配置 | `llm_finetune/config.py` | Prompt模板、LoRA参数 |
| LLM 训练 | `llm_finetune/trainer.py` | QLoRA微调 |
| 推理脚本 | `inference/run_inference.py` | Trie约束解码 |
| 约束解码 | `inference/constrained_decoding.py` | TrieConstrainedLogitsProcessor |
| Yelp数据 | `dataset/yelp/processed/Philadelphia/` | Philadelphia子集 |
| 语义ID输出 | `output/sid/PA_main_city/semantic_ids_v2.json` | 语义ID结果 |
| LoRA权重 | `checkpoints/llm_sid/` | 微调后的LoRA adapters |
