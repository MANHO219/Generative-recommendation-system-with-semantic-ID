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
    - `test_samples.json` 将变为“每用户仅一条最后目标”测试集。
    - `train/val` 仍来自滑窗样本（去除了每用户最后一条，避免与测试目标重叠）。

### 直接训练（含严格闭包）
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
/mnt/data/liuwei/anaconda3/envs/ywh/bin/python \
/mnt/data/liuwei/yewenhao/main/llm_finetune/trainer.py \
    --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids.json \
    --min_user_interactions 5 \
    --strict_kcore --k_core 5 \
    --force_rebuild_cache
```

- 训练入口和 `build_data_only.py` 现在共用同一 `prepare_datasets()`，严格闭包逻辑一致。

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
