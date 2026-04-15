# 推理模块说明（Trie 约束解码）

本目录提供基于 Trie 的约束解码推理能力：只允许模型输出在 `semantic_ids` 字典中存在的合法 SID。

## 文件说明

- `trie.py`：Token 级 Trie 结构。
- `constrained_decoding.py`：约束解码用 `LogitsProcessor`。
- `run_inference.py`：当前主推理脚本（支持单样本与批量评测）。
- `hr_eval.py`：对预测结果计算 `HR@1/HR@K/MRR@K/NDCG@K`。
- `test_trie.py`：Trie 最小功能测试。

## 当前支持的推理命令

> 先进入项目根目录 `main/`，推荐使用模块方式运行。

### 1）查看参数

```bash
cd /mnt/data/liuwei/yewenhao/main
python -m inference.run_inference --help
```

`run_inference.py` 当前支持以下参数：
- `--model_path`
- `--base_model_path`（可选，LoRA 适配器缺少 base 信息时可显式指定）
- `--semantic_ids_path`
- `--sample_json`（可选）
- `--eval_samples_path`（可选，批量评测输入）
- `--eval_limit`（可选）
- `--print_examples`（可选）
- `--log_every`（可选）
- `--eval_report_path`（可选）
- `--eval_predictions_path`（可选，保存逐样本预测含 Top-K 候选）
- `--max_new_tokens`（可选）
- `--num_beams`（可选）
- `--top_k`（可选）
- `--temperature`（可选）

### 2）单样本推理（你的 LoRA checkpoint）

```bash
cd /mnt/data/liuwei/yewenhao/main
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 /mnt/data/liuwei/anaconda3/envs/ywh/bin/python -m inference.run_inference \
  --model_path /mnt/data/liuwei/yewenhao/main/checkpoints/llm_sid/final_model \
  --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids_v2.json \
  --sample_json /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_samples.json \
  --num_beams 5 --max_new_tokens 30 --temperature 0
```

说明：
- `--sample_json` 可传单条 JSON（字典）或 JSON 列表（会自动使用第一条样本）。

### 3）不传样本时的快速测试

```bash
cd /mnt/data/liuwei/yewenhao/main
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 /mnt/data/liuwei/anaconda3/envs/ywh/bin/python -m inference.run_inference \
  --model_path /mnt/data/liuwei/yewenhao/main/checkpoints/llm_sid/final_model \
  --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids_v2.json
```

### 4）批量评测（方案 A，主脚本内完成）

```bash
cd /mnt/data/liuwei/yewenhao/main
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 /mnt/data/liuwei/anaconda3/envs/ywh/bin/python -m inference.run_inference \
  --model_path /mnt/data/liuwei/yewenhao/main/checkpoints/llm_sid/final_model \
  --base_model_path /mnt/data/liuwei/yewenhao/main/models/JunHowie/Qwen3-8B-Instruct \
  --semantic_ids_path /mnt/data/liuwei/yewenhao/main/output/sid/PA_main_city/semantic_ids_v2.json \
  --eval_samples_path /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_samples.json \
  --eval_limit 500 \
  --num_beams 5 \
  --top_k 5 \
  --max_new_tokens 30 \
  --temperature 0 \
  --print_examples 20 \
  --log_every 50 \
  --eval_report_path /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final.json \
  --eval_predictions_path /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final_topk.jsonl
```

### 5）真 Top-K 评估（推荐）

先用上面的 `--eval_predictions_path` 导出逐样本候选，然后评估该文件：

```bash
python /mnt/data/liuwei/yewenhao/main/inference/hr_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final_topk.jsonl \
  --top_k 5 \
  --save_path /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final_topk_hr.json

python /mnt/data/liuwei/yewenhao/main/inference/bucket_hit_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final_topk.jsonl \
  --top_k 5 \
  --prompt_field prompt \
  --save_path /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final_topk_bucket.json
```

说明：
- 真 Top-K 要求预测文件中存在列表字段（如 `candidates`）。
- `generated_predictions.jsonl` 通常只有单条 `predict`，此时 `HR@5` 退化为 `HR@1`。

## 结果评估命令（HR 指标）

使用 `hr_eval.py` 对预测文件（`.jsonl` 或 `.json`）进行离线评估：

```bash
python /mnt/data/liuwei/yewenhao/main/inference/hr_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/output/llamafactory/gnpr_qwen3_4b_lora_predict/generated_predictions.jsonl \
  --top_k 5 \
  --save_path /mnt/data/liuwei/yewenhao/main/output/llamafactory/gnpr_qwen3_4b_lora_predict/hr_metrics_k5.json

python /mnt/data/liuwei/yewenhao/main/inference/hr_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/generated_predictions.jsonl \
  --top_k 20 \
  --save_path /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/hr_metrics_k5.json
```

输出指标包括：
- `hr@1`
- `hr@k`
- `mrr@k`
- `ndcg@k`
- `single_prediction_ratio`

## Prompt 长度统计与分桶评估

### 1）统计 Prompt 历史长度分布

使用 `prompt_history_stats.py` 统计 `input/prompt` 中历史访问条数分布（含分位数、频次和分桶占比）：

```bash
python /mnt/data/liuwei/yewenhao/main/inference/prompt_history_stats.py \
  --prompt_file /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_2k.json \
  --prompt_field input \
  --save_path /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_2k_hist_stats.json
```

### 2）按历史长度分桶评估命中率

使用 `bucket_hit_eval.py` 对预测结果按历史长度分桶统计 `HR@1/HR@K/MRR@K`：

```bash
python /mnt/data/liuwei/yewenhao/main/inference/bucket_hit_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/generated_predictions.jsonl \
  --top_k 5 \
  --prompt_field prompt \
  --save_path /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/bucket_hit_eval.json
```

说明：
- `--prompt_field` 默认是 `prompt`，如果你的预测文件字段是 `input`，可改为 `--prompt_field input`。
- 分桶规则可通过 `--bins` 指定，如：`1-5,6-10,11-20,21-30,31-40,41-50`。

## 注意事项

- 为保证评测口径一致，请固定 `semantic_ids_path` 与训练时一致。
