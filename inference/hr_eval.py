import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ANGLE_SID_PATTERN = re.compile(r"(?:<[a-d]_\d+>){3,4}")
DASH_SID_PATTERN = re.compile(r"\d+-\d+-\d+(?:(?:<d_\d+>)|\[[^\]]+\])?")


def is_angle_bracket_sid(value: str) -> bool:
    return bool(re.fullmatch(r"(?:<[a-d]_\d+>){3,4}", value.strip()))


def to_angle_bracket_sid(value: str) -> str:
    sid = value.strip()
    if is_angle_bracket_sid(sid):
        return sid

    match = re.fullmatch(r"(\d+)-(\d+)-(\d+)(?:(?:<d_(\d+)>)|\[([^\]]+)\])?", sid)
    if not match:
        raise ValueError(f"Unsupported SID format: {value}")

    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    base_sid = "".join(f"<{label}_{item}>" for label, item in zip(["a", "b", "c"], values))
    d_suffix = match.group(4)
    if d_suffix is not None:
        return f"{base_sid}<d_{int(d_suffix)}>"

    disambig = match.group(5)
    if disambig is None:
        return base_sid

    trailing_index = re.search(r"_(\d+)$", disambig)
    if trailing_index:
        return f"{base_sid}<d_{int(trailing_index.group(1))}>"

    pure_number = re.fullmatch(r"\d+", disambig)
    if pure_number:
        return f"{base_sid}<d_{int(disambig)}>"

    return f"{base_sid}<d_0>"


def normalize_sid(candidate: str) -> Optional[str]:
    text = candidate.strip()
    if not text:
        return None

    token_matches = re.findall(r"<[a-d]_\d+>", text)
    if len(token_matches) >= 3:
        return "".join(token_matches[:4]) if len(token_matches) >= 4 else "".join(token_matches[:3])

    angle_match = ANGLE_SID_PATTERN.search(text)
    if angle_match:
        return angle_match.group(0)

    dash_match = DASH_SID_PATTERN.search(text)
    if dash_match:
        try:
            return to_angle_bracket_sid(dash_match.group(0))
        except ValueError:
            return None

    try:
        return to_angle_bracket_sid(text)
    except ValueError:
        return None


def extract_sid_candidates(text: str) -> List[str]:
    raw_candidates = ANGLE_SID_PATTERN.findall(text)
    if not raw_candidates:
        dash_candidates = DASH_SID_PATTERN.findall(text)
        raw_candidates = [to_angle_bracket_sid(item) for item in dash_candidates]

    deduped: List[str] = []
    seen = set()
    for item in raw_candidates:
        norm = normalize_sid(item)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if path.suffix.lower() == ".json":
        loaded = json.loads(content)
        if isinstance(loaded, list):
            return loaded
        raise ValueError("JSON file must contain a list of records.")

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def pick_first_str(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_candidates(record: Dict[str, Any]) -> List[str]:
    list_keys = ["candidates", "topk", "top_k", "top_k_predictions", "predict_topk"]
    for key in list_keys:
        value = record.get(key)
        if isinstance(value, list):
            parsed: List[str] = []
            seen = set()
            for item in value:
                if not isinstance(item, str):
                    continue
                normalized = normalize_sid(item)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    parsed.append(normalized)
            if parsed:
                return parsed

    pred_text = pick_first_str(record, ["predict", "prediction", "pred", "text"])
    if pred_text:
        extracted = extract_sid_candidates(pred_text)
        if extracted:
            return extracted
        normalized_pred = normalize_sid(pred_text)
        if normalized_pred:
            return [normalized_pred]
    return []


def parse_label(record: Dict[str, Any]) -> Optional[str]:
    label_text = pick_first_str(record, ["label", "output", "target_sid", "gt", "ground_truth"])
    if not label_text:
        return None

    normalized = normalize_sid(label_text)
    if normalized:
        return normalized

    extracted = extract_sid_candidates(label_text)
    if extracted:
        return extracted[0]
    return None


def evaluate(records: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    total = 0
    skipped = 0
    hr1 = 0
    hrk = 0
    mrr = 0.0
    ndcg = 0.0
    used_single_prediction_only = 0

    for row in records:
        label = parse_label(row)
        candidates = parse_candidates(row)
        if label is None or not candidates:
            skipped += 1
            continue

        total += 1
        if len(candidates) == 1:
            used_single_prediction_only += 1

        if candidates[0] == label:
            hr1 += 1

        topk = candidates[:top_k]
        if label in topk:
            hrk += 1
            rank = topk.index(label) + 1
            mrr += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)

    if total == 0:
        raise ValueError("No valid samples after parsing labels/candidates.")

    # Recall@K: for single-label scenario, same as HR@K (1 if label in top-k, else 0)
    recall = hrk / total

    return {
        "total": total,
        "skipped": skipped,
        "top_k": top_k,
        "hr@1": hr1 / total,
        f"hr@{top_k}": hrk / total,
        f"recall@{top_k}": recall,
        f"mrr@{top_k}": mrr / total,
        f"ndcg@{top_k}": ndcg / total,
        "single_prediction_ratio": used_single_prediction_only / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HR@K/MRR/NDCG from prediction files.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to prediction file (.jsonl or .json)")
    parser.add_argument("--top_k", type=int, default=5, help="K for HR@K/MRR@K/NDCG@K")
    parser.add_argument("--save_path", type=str, default="", help="Optional path to save metrics JSON")
    args = parser.parse_args()

    records = read_json_or_jsonl(Path(args.pred_file))
    metrics = evaluate(records, args.top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
