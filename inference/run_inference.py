import argparse
import json
import re
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from peft import PeftModel

try:
    from .constrained_decoding import TrieConstrainedLogitsProcessor
    from .trie import TokenTrie
except ImportError:
    from constrained_decoding import TrieConstrainedLogitsProcessor
    from trie import TokenTrie

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_finetune.config import PROMPT_TEMPLATE


def is_angle_bracket_sid(sid: str) -> bool:
    return bool(re.fullmatch(r'(<[a-d]_\d+>){3,4}', sid.strip()))


def to_angle_bracket_sid(sid: str) -> str:
    sid = sid.strip()
    if is_angle_bracket_sid(sid):
        return sid
    match = re.fullmatch(r'(\d+)-(\d+)-(\d+)(?:(?:<d_(\d+)>)|\[([^\]]+)\])?', sid)
    if not match:
        raise ValueError(f'Unsupported SID format: {sid}')
    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    base_sid = ''.join(f'<{label}_{value}>' for label, value in zip(['a', 'b', 'c'], values))
    d_suffix = match.group(4)
    if d_suffix is not None:
        return f'{base_sid}<d_{int(d_suffix)}>'
    disambig = match.group(5)
    if disambig is None:
        return base_sid
    trailing_index = re.search(r'_(\d+)$', disambig)
    if trailing_index:
        return f'{base_sid}<d_{int(trailing_index.group(1))}>'
    pure_number = re.fullmatch(r'\d+', disambig)
    if pure_number:
        return f'{base_sid}<d_{int(disambig)}>'
    return f'{base_sid}<d_0>'


def build_trie(tokenizer, semantic_ids: List[str]) -> TokenTrie:
    trie = TokenTrie()
    for sid in semantic_ids:
        token_ids = tokenizer.encode(sid, add_special_tokens=False)
        trie.add(token_ids)
    return trie


def load_semantic_ids(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    return list({to_angle_bracket_sid(sid) for sid in data.values()})


def format_prompt(sample: Dict[str, Any]) -> str:
    if {'instruction', 'input', 'output'}.issubset(sample.keys()):
        return sample['input']

    sid_template = PROMPT_TEMPLATE['sid_time_history']
    user_id = sample.get('user_id', 'unknown_user')
    target = sample.get('target', {})
    target_time = target.get('date', sample.get('target_time'))
    if isinstance(target_time, str):
        target_time = datetime.fromisoformat(target_time)
    elif not isinstance(target_time, datetime):
        raise ValueError('sample.target.date or sample.target_time is required')

    formatted_items = []
    for visit in sample.get('history', []):
        visit_time = visit.get('date') or visit.get('time')
        sid = visit.get('sid')
        if sid is None and 'business_id' in visit and 'business_sid_map' in sample:
            sid = sample['business_sid_map'].get(visit['business_id'])
        if sid is None:
            continue
        sid = to_angle_bracket_sid(sid)
        if isinstance(visit_time, str):
            visit_time = datetime.fromisoformat(visit_time)
        if not isinstance(visit_time, datetime):
            continue
        formatted_items.append(
            sid_template['history_item'].format(
                time=visit_time.strftime('%Y-%m-%d %H:%M:%S'),
                sid=sid,
            )
        )

    history_text = ', '.join(formatted_items)
    return (
        f"{sid_template['history_prefix'].format(user_id=user_id)}{history_text}.\n"
        f"{sid_template['query_suffix'].format(target_time=target_time.strftime('%Y-%m-%d %H:%M:%S'), user_id=user_id)}"
    )


def apply_chat_template_no_think(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {
        'tokenize': False,
        'add_generation_prompt': add_generation_prompt,
    }
    signature = inspect.signature(tokenizer.apply_chat_template)
    if 'enable_thinking' in signature.parameters:
        kwargs['enable_thinking'] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    rendered = re.sub(r'<think>[\s\S]*?</think>\s*', '', rendered)
    rendered = rendered.replace('<think>', '').replace('</think>', '')
    return rendered


def build_chat_prompt(tokenizer, sample: Dict[str, Any]) -> str:
    user_prompt = format_prompt(sample)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": PROMPT_TEMPLATE["system"]},
            {"role": "user", "content": user_prompt},
        ]
        return apply_chat_template_no_think(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
    return user_prompt


def normalize_sample_payload(sample_payload: Any) -> Dict[str, Any]:
    if isinstance(sample_payload, dict):
        return sample_payload

    if isinstance(sample_payload, list):
        if not sample_payload:
            raise ValueError("sample_json is an empty list.")
        first_item = sample_payload[0]
        if not isinstance(first_item, dict):
            raise ValueError("sample_json list items must be JSON objects.")
        print("[Info] sample_json contains a list; using the first sample.")
        return first_item

    raise ValueError("sample_json must be a JSON object or a list of JSON objects.")


def normalize_sid_text(value: str) -> str:
    text = value.strip()
    if not text:
        return text

    token_matches = re.findall(r'<[a-d]_\d+>', text)
    if len(token_matches) >= 3:
        return ''.join(token_matches[:4]) if len(token_matches) >= 4 else ''.join(token_matches[:3])

    angle_match = re.search(r'(<[a-d]_\d+>){3,4}', text)
    if angle_match:
        return angle_match.group(0)

    dash_match = re.search(r'(\d+-\d+-\d+(?:(?:<d_\d+>)|\[[^\]]+\])?)', text)
    if dash_match:
        try:
            return to_angle_bracket_sid(dash_match.group(1))
        except ValueError:
            return text

    try:
        return to_angle_bracket_sid(text)
    except ValueError:
        return text


def load_model_and_tokenizer(model_path: str, base_model_path: Optional[str] = None):
    model_dir = Path(model_path)
    is_adapter = (model_dir / "adapter_config.json").exists() and not (model_dir / "config.json").exists()

    if is_adapter:
        with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as file:
            adapter_config = json.load(file)
        resolved_base_model = base_model_path or adapter_config.get("base_model_name_or_path")
        if not resolved_base_model:
            raise ValueError("Adapter checkpoint requires --base_model_path or valid base_model_name_or_path.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_sid_candidates(model, tokenizer, sample: Dict[str, Any], trie: TokenTrie, args) -> List[str]:
    prompt = build_chat_prompt(tokenizer, sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prefix_len = inputs.input_ids.shape[1]

    processor = TrieConstrainedLogitsProcessor(
        trie=trie,
        prefix_length=prefix_len,
        eos_token_id=tokenizer.eos_token_id,
        allow_early_stop=True,
    )

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_beams,
        "do_sample": args.temperature > 0,
        "logits_processor": LogitsProcessorList([processor]),
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    else:
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        generation_kwargs["top_k"] = None

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    candidates = []
    seen = set()
    for row_index in range(outputs.shape[0]):
        pred = tokenizer.decode(outputs[row_index][prefix_len:], skip_special_tokens=True).strip()
        pred = normalize_sid_text(pred)
        if pred and pred not in seen:
            seen.add(pred)
            candidates.append(pred)
    return candidates


def run_batch_eval(model, tokenizer, trie: TokenTrie, args, semantic_sid_set: set[str]) -> None:
    with open(args.eval_samples_path, "r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, list):
        raise ValueError("--eval_samples_path must point to a JSON list.")

    if args.eval_limit and args.eval_limit > 0:
        samples = samples[: args.eval_limit]

    total = len(samples)
    if total == 0:
        raise ValueError("No eval samples found.")

    exact_match = 0
    hit_at_k = 0
    recall_at_k = 0
    mrr_sum = 0.0
    valid_sid = 0
    latencies = []
    top_k = max(1, args.top_k)
    prediction_rows = []

    for index, sample in enumerate(samples, 1):
        start_time = time.time()
        candidates = generate_sid_candidates(model, tokenizer, sample, trie, args)
        latencies.append((time.time() - start_time) * 1000)

        top1 = candidates[0] if candidates else ""
        topk = candidates[:top_k]

        raw_gt = str(sample.get("target_sid", sample.get("output", ""))).strip()
        gt = normalize_sid_text(raw_gt)

        if top1 == gt:
            exact_match += 1
        if top1 in semantic_sid_set:
            valid_sid += 1
        if gt in topk:
            hit_at_k += 1
            recall_at_k += 1
            rank = topk.index(gt) + 1
            mrr_sum += 1.0 / rank

        if args.eval_predictions_path:
            prompt_text = ""
            if isinstance(sample.get("input"), str):
                prompt_text = sample["input"]
            elif isinstance(sample.get("prompt"), str):
                prompt_text = sample["prompt"]
            prediction_rows.append(
                {
                    "prompt": prompt_text,
                    "predict": top1,
                    "label": gt,
                    "candidates": topk,
                }
            )

        if index <= args.print_examples:
            print(f"[Example {index}] top1={top1} | gt={gt} | exact={top1 == gt} | hit@{top_k}={gt in topk}")
        if index % args.log_every == 0:
            print(f"Processed {index}/{total}")

    print("\n=== Batch Evaluation Summary ===")
    print(f"total_samples: {total}")
    print(f"exact_match: {exact_match}")
    print(f"exact_match_rate: {exact_match / total:.4f}")
    print(f"hit@{top_k}: {hit_at_k / total:.4f}")
    print(f"recall@{top_k}: {recall_at_k / total:.4f}")
    print(f"mrr@{top_k}: {mrr_sum / total:.4f}")
    print(f"valid_sid_rate: {valid_sid / total:.4f}")
    print(f"avg_latency_ms: {sum(latencies) / len(latencies):.2f}")

    summary = {
        "total_samples": total,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / total,
        f"hit@{top_k}": hit_at_k / total,
        f"recall@{top_k}": recall_at_k / total,
        f"mrr@{top_k}": mrr_sum / total,
        "valid_sid_rate": valid_sid / total,
        "avg_latency_ms": sum(latencies) / len(latencies),
    }

    if args.eval_report_path:
        report_path = Path(args.eval_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_report: {report_path}")

    if args.eval_predictions_path:
        predictions_path = Path(args.eval_predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as file:
            for row in prediction_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved_predictions: {predictions_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--semantic_ids_path", type=str, required=True)
    parser.add_argument("--sample_json", type=str, default=None)
    parser.add_argument("--eval_samples_path", type=str, default=None)
    parser.add_argument("--eval_limit", type=int, default=100)
    parser.add_argument("--print_examples", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_report_path", type=str, default=None)
    parser.add_argument("--eval_predictions_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_path)

    semantic_ids = load_semantic_ids(args.semantic_ids_path)
    trie = build_trie(tokenizer, semantic_ids)
    semantic_sid_set = set(semantic_ids)

    if args.eval_samples_path:
        run_batch_eval(model, tokenizer, trie, args, semantic_sid_set)
        return

    if args.sample_json:
        sample_payload = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
        sample = normalize_sample_payload(sample_payload)
    else:
        sample = {
            "user_id": "demo_user_123456",
            "history": [
                {"sid": "<a_31><b_73><c_58>", "time": "2024-11-01T10:15:00"},
                {"sid": "<a_66><b_122><c_25>", "time": "2024-11-03T19:20:00"},
                {"sid": "<a_74><b_51><c_62>", "time": "2024-11-05T08:02:00"},
            ],
            "target": {"date": "2024-11-06T12:00:00"},
        }

    candidates = generate_sid_candidates(model, tokenizer, sample, trie, args)
    print(candidates[0] if candidates else "")


if __name__ == "__main__":
    main()
