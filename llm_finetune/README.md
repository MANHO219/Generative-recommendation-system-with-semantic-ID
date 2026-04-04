# LLM 微调模块

## 训练命令

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
模型接收用户历史访问序列和时空上下文，生成目标 POI 的三层 Semantic ID。

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

### 输入（ChatML 格式，三段式）

```
<|im_start|>system
You are an intelligent POI recommendation assistant. ...<|im_end|>
<|im_start|>user
### User Profile:
- User ID: abc12345
- Active Level: 150 reviews
- Average Rating: 4.2 stars
- Favorite Categories: Restaurants, Coffee & Tea, Bars

### Spatiotemporal Context:
- Current Location: Plus Code 3316+XX
- Time: Friday, Evening (19:00)
- Day Type: Weekday

### Visit History (Recent 5 visits):
1. Starbucks (Coffee & Tea) - Rated 4.0 stars - 2024-01-15
2. McDonald's (Fast Food) - Rated 3.5 stars - 2024-01-14
...

Based on the above information, predict the Semantic ID of the next POI ...
Output only the Semantic ID in format: XX-XX-XX[GG]<|im_end|>
<|im_start|>assistant
```

### 输出

```
12-34-56[GQ]
```

三层 RQ-VAE Semantic ID，基础格式为 `XX-XX-XX`；
若该 SID 存在冲突，追加后缀为 `XX-XX-XX[GG]` 或 `XX-XX-XX[GG_n]`。

---

## 数据流

```
Yelp 原始数据（business_poi / user_active / review_poi）
    ↓ DatasetBuilder.build_and_save()
滑窗样本（history + target + target_sid）
    ↓ random.seed(42) shuffle → 80/10/10 切分
output/dataset_cache/{train,val,test}_samples.json   ← 缓存文件
    ↓ LLMFinetuneDataset.__getitem__() → format_instruction()
ChatML 格式样本
    ↓ build_hf_dataset() → HuggingFace Dataset（text 字段）
SFTTrainer
```

**注意**：`target_sid` 在构建时从 `output/semantic_ids.json` 写入缓存。
更换 Semantic ID 后必须删除缓存，否则训练标签与推理 Trie 不一致。

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
- 首次运行或缓存不存在时自动构建（耗时较长）
- 缓存文件损坏时自动删除并重建
- **更换 `semantic_ids.json` 后必须手动删除缓存**，否则 `target_sid` 标签错误

---

## 注意事项

- **显存**：单卡 24GB 可跑 batch_size=6，低于 16GB 需调小 batch
- **Plus Code**：当前为简化坐标区间版本，精度有限，不影响训练
- **导入路径**：`inference/constrained_decoding.py` 中使用 `from inference.trie import TokenTrie`，从项目根目录启动才能正确解析
