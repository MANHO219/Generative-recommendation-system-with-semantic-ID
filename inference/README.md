# Inference with Trie-Constrained Decoding

This folder provides constrained decoding based on a Trie built from valid Semantic IDs. The constraint ensures the model only generates IDs that exist in your dataset.

## Files
- `trie.py`: Token-level trie structure.
- `constrained_decoding.py`: LogitsProcessor that masks invalid tokens.
- `run_inference.py`: CLI for constrained inference.
- `test_trie.py`: Minimal trie test.

## Quick Start

```bash
python /root/autodl-tmp/main/inference/test_trie.py
```

```bash
python /root/autodl-tmp/main/inference/run_inference.py \
  --model_path /root/autodl-tmp/main/models/JunHowie/Qwen3-8B-Instruct \
  --semantic_ids_path /root/autodl-tmp/main/output/semantic_ids.json
```

## Using a Custom Prompt

```bash
python /root/autodl-tmp/main/inference/run_inference.py \
  --model_path /root/autodl-tmp/main/models/JunHowie/Qwen3-8B-Instruct \
  --semantic_ids_path /root/autodl-tmp/main/output/semantic_ids.json \
  --prompt_file /root/autodl-tmp/main/inference/prompt.txt
```

The constrained decoder will only output valid Semantic IDs from `semantic_ids.json`.
