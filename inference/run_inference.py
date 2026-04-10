import argparse
import json
import re
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

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
    match = re.fullmatch(r'(\d+)-(\d+)-(\d+)(?:\[([^\]]+)\])?', sid)
    if not match:
        raise ValueError(f'Unsupported SID format: {sid}')
    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    base_sid = ''.join(f'<{label}_{value}>' for label, value in zip(['a', 'b', 'c'], values))
    disambig = match.group(4)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_ids_path", type=str, required=True)
    parser.add_argument("--sample_json", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.sample_json:
        sample = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
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

    prompt = build_chat_prompt(tokenizer, sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    semantic_ids = load_semantic_ids(args.semantic_ids_path)
    trie = build_trie(tokenizer, semantic_ids)

    prefix_len = inputs.input_ids.shape[1]
    processor = TrieConstrainedLogitsProcessor(
        trie=trie,
        prefix_length=prefix_len,
        eos_token_id=tokenizer.eos_token_id,
        allow_early_stop=True,
    )

    logits_processors = LogitsProcessorList([processor])

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        logits_processor=logits_processors,
    )

    generated = tokenizer.decode(outputs[0][prefix_len:], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
