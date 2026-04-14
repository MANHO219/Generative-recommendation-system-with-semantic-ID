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
  --eval_report_path /mnt/data/liuwei/yewenhao/main/logs/inference/eval_final.json
```

## 结果评估命令（HR 指标）

使用 `hr_eval.py` 对预测文件（`.jsonl` 或 `.json`）进行离线评估：

```bash
python /mnt/data/liuwei/yewenhao/main/inference/hr_eval.py \
  --pred_file /mnt/data/liuwei/yewenhao/main/output/llamafactory/gnpr_qwen3_4b_lora_predict/generated_predictions.jsonl \
  --top_k 5 \
  --save_path /mnt/data/liuwei/yewenhao/main/output/llamafactory/gnpr_qwen3_4b_lora_predict/hr_metrics_k5.json
```

输出指标包括：
- `hr@1`
- `hr@k`
- `mrr@k`
- `ndcg@k`
- `single_prediction_ratio`

## 注意事项

- 为保证评测口径一致，请固定 `semantic_ids_path` 与训练时一致。
