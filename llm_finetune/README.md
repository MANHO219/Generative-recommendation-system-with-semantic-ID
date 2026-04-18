# LLM 微调模块

## 训练命令

## 仅构建数据缓存（不启动训练）

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/build_data_only.py \
    --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids.json \
    --min_user_interactions 5 \
    --test_mode sliding \
    --strict_kcore --k_core 5 \
    --cache_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
    --prompt_export_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
    --force_rebuild
```

- 该命令只会调用 `prepare_datasets()`，生成：
    - `train/val/test_samples.json`
    - `train/val/test_prompts.json`（默认导出，可用 `--no_export_prompts` 关闭）
- 默认 `--test_mode sliding`：训练/验证/测试都按滑动窗口构建；同时会额外导出公平对比用的 `test_last_item_samples.json` 与 `test_last_item_prompts.json`。
- `--min_user_interactions` 会在构建时显式过滤用户（按“可映射到 SID 的有效 visits 数”计数），推荐设为 `5` 以对齐 k-core 口径。
- `--strict_kcore --k_core 5` 会在构建前调用 `semantic_id/kcore.py` 的迭代闭包过滤逻辑，确保用户/商铺双向闭包一致。
- 更换新的 codebook/SID 后，建议加 `--force_rebuild` 强制重建缓存。

### 两种预处理流程

通过 `--preprocess_pipeline` 选择数据预处理流程：

| 流程 | 说明 |
|------|------|
| `legacy`（默认） | 原始滑动窗口方式，min_user_interactions 控制用户过滤 |
| `yelp_session` | NYC 风格会话切分，含全局时间切分、24h 孤立点剔除、会话切分、冷启动消除 |

### Yelp Session 流程参数（preprocess_pipeline=yelp_session）

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/build_data_only.py \
    --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids.json \
    --preprocess_pipeline yelp_session \
    --session_enable_filter_low_frequency \
    --session_min_poi_freq 10 \
    --session_min_user_freq 10 \
    --session_time_interval_min 1440 \
    --no_session_remove_isolated_24h \
    --no_session_ignore_singleton_sessions \
    --no_session_remove_unseen_user_poi \
    --strict_kcore --k_core 5 \
    --cache_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_yelp_session \
    --prompt_export_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_yelp_session \
    --force_rebuild
```

**会话切分时间间隔分析**（基于 k-core 过滤后的数据）：

| 统计量 | 值 |
|--------|-----|
| 样本总数（≥2 条历史） | 463,322 |
| 相邻间隔总数 | 9,529,257 |
| 整体平均间隔（小时） | 664.6 |
| 整体中位间隔（小时） | 47.2 |
| 每样本平均间隔（小时） | 1487.1 |
| 每样本中位间隔（小时） | 443.3 |
| 最小间隔（小时） | 0.0 |
| 最大间隔（小时） | 105,773.4 |

- **1440 分钟（24 小时）** 作为会话切分阈值：大多数真实访问间隔在 47 小时（中位数）以内，超过 24 小时的间隔被视为跨会话的独立访问序列。
- 中位间隔 47 小时说明用户通常在 2 天内会有下一次访问，但存在长尾分布（平均 665 小时 ≈ 28 天），表明确实需要时间感知的会话切分策略。

**Yelp Session Pipeline 处理步骤**：

1. `filter_low_frequency`：过滤 review 中访问次数 ≤10 的 POI 和活跃度 ≤10 的用户
2. `split_by_global_time`：全局时间顺序切分 80%/10%/10%（train/val/test）
3. `remove_isolated_24h`：剔除前后时间间隔均超过 24 小时的孤立访问记录
4. `build_pseudo_sessions`：按时间间隙（默认 1440 分钟）切分会话
5. `ignore_singleton_sessions`：剔除仅包含单个访问点的会话
6. `remove_unseen_user_poi`：确保 val/test 中的用户和 POI 在 train 中均出现过（消除冷启动）

**注意**：`--strict_kcore --k_core 5` 会在 Session Pipeline 之前执行独立的数据闭包过滤，与 `session_enable_filter_low_frequency` 串联生效。

### 公平对齐口径（每用户最后一次目标）

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/build_data_only.py \
        --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids.json \
        --min_user_interactions 5 \
        --test_mode last_item \
        --strict_kcore --k_core 5 \
        --cache_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
        --prompt_export_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
        --force_rebuild
```

- `--test_mode last_item` 下：
    - `test_samples.json` 将变为”每用户仅一条最后目标”测试集。
    - `train/val` 仍来自滑窗样本（去除了每用户最后一条，避免与测试目标重叠）。
- `--preprocess_pipeline=yelp_session` 时，`test_mode` 参数被忽略，测试集直接来自 Session Pipeline 的全局时间切分。

### 提取子集（快速实验）

如果你已经构建好了完整缓存，可以二次抽样出 `train=10000`、`val=2000`、`test=2000`：

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/extract_subset.py \
    --source_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
    --target_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore_10k2k2k \
    --train_n 50000 \
    --val_n 10000 \
    --test_n 10000
```

如果测试要采用 STAN 对齐口径（每用户最后一次），可把测试来源切到 `test_last_item`：

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/extract_subset.py \
    --source_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore \
    --target_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_kcore_10k2k2k_last \
    --train_n 10000 \
    --val_n 2000 \
    --test_n 2000 \
    --test_source test_last_item
```

两种抽取模式：

| 模式 | 参数 | 行为 |
|------|------|------|
| 按样本数 | `--train_n` | 随机选 N 条样本 |
| 按用户比例 | `--user_fraction` | 随机选 X% 的用户，保留其所有样本 |

**按用户比例抽取示例**（保留每个用户数据完整性）：

```bash
PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/extract_subset.py \
    --source_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_yelp_session \
    --target_dir /mnt/data/liuwei/yewenhao/main/output/dataset_cache_yelp_session_frac \
    --user_fraction 0.1
```

- `--user_fraction 0.1` 随机选取 10% 的用户，保留这些用户的所有样本
- `--val_user_fraction 0.2` 可独立设置 val 集的用户比例（默认同 `--user_fraction`）
- 与 `--train_n` 等参数互斥，不能同时使用
- 输出包含：`train/val/test_{samples,prompts}.json` 与 `subset_manifest.json`
- 默认随机抽样（`seed=42`）；若想按原顺序截取前 N 条，增加 `--no_shuffle`

### 直接训练（含严格闭包）
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/trainer.py \
    --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids.json \
    --min_user_interactions 5 \
    --strict_kcore --k_core 5 \
    --preprocess_pipeline yelp_session \
    --session_enable_filter_low_frequency \
    --session_min_poi_freq 10 \
    --session_min_user_freq 10 \
    --force_rebuild_cache
```

- 训练入口和 `build_data_only.py` 共用同一 `prepare_datasets()`，参数同步透传。
- `--preprocess_pipeline yelp_session` 启用 NYC 风格会话 pipeline，与 `strict_kcore` 串联生效。

### 单卡训练
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/trainer.py \
|& tee -a /mnt/data/liuwei/yewenhao/main/logs/llm_finetune/train_single_$(date +%F_%H-%M-%S).log
```

### 多卡训练（accelerate）
```bash
CUDA_VISIBLE_DEVICES=0,1,3 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/accelerate launch \
/mnt/data/liuwei/yewenhao/main/llm_finetune/trainer.py \
|& tee -a /mnt/data/liuwei/yewenhao/main/logs/llm_finetune/train_multi_$(date +%F_%H-%M-%S).log
```

### 重建数据缓存（更换 semantic_ids.json 后必须执行）
```bash
rm -rf /mnt/data/liuwei/yewenhao/main/output/dataset_cache/
```
删除后重新启动训练，程序会自动重建缓存。

---

## 模块概述

基于 QLoRA 的 Qwen3-8B-Instruct 指令微调，用于 POI Semantic ID 序列预测。
默认主路径采用 **SID+时间历史**（GNPR/V1 风格），模型接收历史 `time + SID` 序列并预测目标 SID。

---

## 文件结构

```
llm_finetune/
├── config.py       # 所有配置（模型、LoRA、训练、数据、Prompt 模板）
├── dataset.py      # 数据集构建与缓存（DatasetBuilder、LLMFinetuneDataset、prepare_datasets）
├── extract_subset.py # 从完整缓存提取固定规模子集（如 10k/2k/2k）
├── trainer.py      # 训练器主体（LLMFinetune：加载模型、训练、评估）
├── requirements.txt
└── README.md
```

---

## 输入输出格式

### 输入（ChatML 格式，三段式，默认主路径）

```
<|im_start|>system
You are a POI next-visit prediction assistant...<|im_end|>
<|im_start|>user
User_abc12345 checkin history: 2024-01-15 10:00:00 visited <a_12><b_34><c_56>,
2024-01-16 18:20:00 visited <a_22><b_18><c_9>.
When 2024-01-17 12:00:00 user_abc12345 is likely to visit:<|im_end|>
<|im_start|>assistant
```

### 输出

```
<a_12><b_34><c_56>
```

默认 SID 格式为 `angle_bracket`，即 `<a_x><b_y><c_z>`（可选 `<d_n>`）。
存在冲突消歧后缀时，输出可为 `<a_x><b_y><c_z><d_n>`，例如 `<a_12><b_34><c_56><d_2>`。

---

## 数据流

```
Yelp 原始数据（business_poi / user_active / review_poi）或 GNPR JSON（instruction/input/output）
    ↓ DatasetBuilder.build_and_save()
标准化样本（history + target + target_sid）
    ↓ `test_mode=sliding`：random.seed(42) shuffle → 80/10/10 切分
    ↓ `test_mode=last_item`：test=每用户最后目标，train/val=其余滑窗样本
output/dataset_cache/{train,val,test}_samples.json   ← 缓存文件
    ↓ LLMFinetuneDataset.__getitem__() → format_instruction()
ChatML 格式样本
    ↓ build_hf_dataset() → HuggingFace Dataset（text 字段）
SFTTrainer
```

**注意**：`target_sid` 在构建时从 `semantic_ids_path` 写入缓存，并统一为 `angle_bracket`。
采用缓存策略 A：模板/schema/SID 字典变化时直接删除缓存并重建。

---

## 模型与训练配置

| 项目 | 值 |
|------|----|
| Base Model | Qwen3-8B-Instruct（本地路径） |
| 量化 | 4-bit NF4 + 双重量化（BitsAndBytes） |
| 计算精度 | bfloat16 |
| Attention | Flash Attention 2 |
| LoRA r / alpha | 16 / 32 |
| LoRA 目标层 | q/k/v/o_proj, gate/up/down_proj |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Per-device batch size | 6（train）/ 6（eval） |
| 梯度累积 | 3 步（有效 batch = 18） |
| Learning Rate | 2e-5（paged_adamw_32bit） |
| Warmup Steps | 100 |
| Max Seq Length | 704（覆盖 100% 样本） |
| Eval / Save Steps | 200 |
| 保留 Checkpoint 数 | 3 + best |
| 序列打包 | packing=True（SFTTrainer） |

---

## 评估指标

训练结束后 `evaluate()` 在测试集上输出两类指标：

### 1. 语言模型损失
- `eval_loss`：Cross-Entropy Loss

### 2. 生成式评测（约束解码，默认取 200 条样本）

| 指标 | 说明 |
|------|------|
| `exact_match_rate` | Top-1 预测与 ground truth 完全匹配 |
| `hit@k` | ground truth 出现在 Top-k 候选中 |
| `recall@k` | 同 hit@k（单目标场景下等价） |
| `mrr@k` | Mean Reciprocal Rank |
| `valid_sid_rate` | Top-1 是合法 SID（在 Trie 中） |
| `avg_latency_ms` | 单样本平均推理耗时（ms） |

生成采用 **Trie 约束解码**（`TrieConstrainedLogitsProcessor`），保证输出一定是合法 SID。

可在 `config.py` 的 `TRAINING_CONFIG` 中调整评测参数：
```python
'eval_generation_samples': 200,   # 评测样本数
'eval_num_beams': 5,              # beam search 宽度
'eval_top_k': 5,                  # top-k 截断
'eval_max_new_tokens': 20,        # 最大生成 token 数
```

---

## 数据缓存说明

- 缓存路径：`output/dataset_cache/{train,val,test}_samples.json`
- Schema 标记：`output/dataset_cache/schema.txt`
- 首次运行或缓存不存在时自动构建（耗时较长）
- 缓存文件损坏时自动删除并重建
- **当 Prompt 模板 / `sid_format_mode` / `semantic_ids_path` 变化时，必须删除缓存并重建**

---

## 注意事项

- **显存**：单卡 24GB 可跑 batch_size=6，低于 16GB 需调小 batch
- **Plus Code**：当前为简化坐标区间版本，精度有限，不影响训练
- **导入路径**：`inference/constrained_decoding.py` 中使用 `from inference.trie import TokenTrie`，从项目根目录启动才能正确解析
