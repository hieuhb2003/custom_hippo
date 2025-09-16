#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

METRIC_KEYS = ["precision", "recall", "f1", "ndcg"]
DEFAULT_OUT_DIR = "/home/hungpv/projects/conversation_magix/eval_results"
ALL_K_SENTINEL = 10**9  # dùng làm key cho @ALL


# ========================= TEXT UTILS =========================
def _normalize(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = (
        t.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u00A0", " ")
    )
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _strip_speakers(t: str) -> str:
    """Bỏ 'User:' / 'Assistant:' nếu có."""
    return re.sub(r"\b(user|assistant)\s*:\s*", "", t, flags=re.I)


def _contains_or_sim(chunk: str, ev: str, contain_threshold: float = 0.85) -> bool:
    """
    Match nhị phân: true nếu evidence là substring của chunk
    hoặc các token của evidence xuất hiện >= contain_threshold trong chunk.
    """
    if ev in chunk:
        return True
    ev_toks = re.findall(r"\w+", ev)
    ch_tokens = set(re.findall(r"\w+", chunk))
    if not ev_toks:
        return False
    inter = sum(1 for w in ev_toks if w in ch_tokens)
    return (inter / len(ev_toks)) >= contain_threshold


# ========================= HELPERS =========================
def _coerce_chunks(item: Dict[str, Any]) -> List[str]:
    """
    Chuẩn hóa item['chunks'] về List[str].
    - Nếu là list[str] -> giữ nguyên
    - Nếu là list[dict] -> ưu tiên các key phổ biến
    """
    raw = item.get("chunks", []) or []
    out: List[str] = []
    for c in raw:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            for key in ("chunk_content", "content", "text", "raw", "value"):
                if key in c and isinstance(c[key], str):
                    out.append(c[key])
                    break
    return out


def _coerce_evidences(item: Dict[str, Any]) -> List[str]:
    ev = item.get("evidences", []) or []
    return [e for e in ev if isinstance(e, str)]


def _has_evidence(item: Dict[str, Any]) -> bool:
    return bool(_coerce_evidences(item))


def _parse_ks(ks_str: str) -> Tuple[Tuple[int, ...], bool]:
    """
    Trả về (ks, use_all). Hỗ trợ 'all' hoặc '0' để dùng toàn bộ chunks.
    """
    parts = [x.strip().lower() for x in ks_str.split(",")]
    use_all = any(x in ("all", "0") for x in parts)
    ks = tuple(sorted({int(x) for x in parts if x.isdigit() and int(x) > 0}))
    return ks, use_all


# ========================= LOW-LEVEL COUNTS =========================
def _evaluate_counts(
    chunks: List[str],
    evidences: List[str],
    k: int | None = None,
    contain_threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Trả về các 'đếm' để có thể cộng dồn micro:
      - tp_evidence: số evidence được cover (để tính recall)
      - retrieved:   số chunk được xét (sau khi cắt k)
      - gold:        tổng #evidence
      - rel_chunks:  số chunk 'relevant' (match >=1 evidence) — để tính precision kiểu IR
      - dcg / idcg:  cho nDCG
    """
    if k is not None:
        chunks = chunks[:k]

    chunks_n = [_normalize(_strip_speakers(c)) for c in chunks]
    evidences_n = [_normalize(_strip_speakers(e)) for e in evidences]

    hit_evs = set()
    rel_chunk_count = 0
    dcg = 0.0

    for rank, ch in enumerate(chunks_n, start=1):
        matched = False
        for ev in evidences_n:
            if _contains_or_sim(ch, ev, contain_threshold=contain_threshold):
                matched = True
                hit_evs.add(ev)
        if matched:
            rel_chunk_count += 1
            dcg += 1.0 / math.log2(rank + 1)

    tp_ev = float(len(hit_evs))
    retrieved = float(len(chunks_n))
    gold = float(len(evidences_n))
    # IDCG dựa trên số chunk liên quan tối đa có thể xếp ở top (<= retrieved),
    # dùng rel_chunk_count để tránh nDCG > 1 khi số chunk liên quan > số evidence
    ideal_hits = int(min(rel_chunk_count, int(retrieved)))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1)) if ideal_hits else 0.0

    return {
        "tp_evidence": tp_ev,
        "retrieved": retrieved,
        "gold": gold,
        "rel_chunks": float(rel_chunk_count),
        "dcg": float(dcg),
        "idcg": float(idcg),
    }


def _counts_to_metrics(cnt: Dict[str, float], precision_mode: str = "ir") -> Dict[str, float]:
    """
    precision_mode:
      - "ir"     : precision = rel_chunks / retrieved  (chuẩn IR, khuyến nghị)
      - "legacy" : precision = tp_evidence / retrieved (giống cách cũ)
    """
    if cnt["retrieved"] > 0:
        if precision_mode == "legacy":
            precision = cnt["tp_evidence"] / cnt["retrieved"]
        else:
            precision = cnt["rel_chunks"] / cnt["retrieved"]
    else:
        precision = 0.0

    recall = (cnt["tp_evidence"] / cnt["gold"]) if cnt["gold"] > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ndcg = (cnt["dcg"] / cnt["idcg"]) if cnt["idcg"] > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "ndcg": ndcg}


# ========================= EVAL CORE =========================
def eval_one_record(
    item: Dict[str, Any],
    ks: Tuple[int, ...],
    use_all: bool,
    contain_threshold: float,
    precision_mode: str,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Trả về:
      - per_k_metrics: metrics theo k (macro đơn vị 1 record)
      - per_k_counts : counts theo k (để cộng dồn làm micro)
    """
    chunks = _coerce_chunks(item)
    evidences = _coerce_evidences(item)

    per_k_metrics: Dict[int, Dict[str, float]] = {}
    per_k_counts: Dict[int, Dict[str, float]] = {}

    # @ALL
    if use_all:
        cnt_all = _evaluate_counts(chunks, evidences, k=None, contain_threshold=contain_threshold)
        per_k_counts[ALL_K_SENTINEL] = cnt_all
        per_k_metrics[ALL_K_SENTINEL] = _counts_to_metrics(cnt_all, precision_mode=precision_mode)

    # @k cụ thể
    for k in ks:
        cnt = _evaluate_counts(chunks, evidences, k=k, contain_threshold=contain_threshold)
        per_k_counts[k] = cnt
        per_k_metrics[k] = _counts_to_metrics(cnt, precision_mode=precision_mode)

    return per_k_metrics, per_k_counts


def eval_dataset(
    dataset: List[Dict[str, Any]],
    ks: Tuple[int, ...] = (3, 5, 10),
    use_all: bool = False,
    contain_threshold: float = 0.85,
    precision_mode: str = "ir",
):
    # macro sums
    macro_sums = {k: {m: 0.0 for m in METRIC_KEYS} for k in ks}
    counts = {k: 0 for k in ks}
    if use_all:
        macro_sums[ALL_K_SENTINEL] = {m: 0.0 for m in METRIC_KEYS}
        counts[ALL_K_SENTINEL] = 0

    failed: List[Tuple[int, str]] = []
    skipped_no_evidence: List[Tuple[int, str]] = []  # (idx, qid)
    per_record = {k: [] for k in ks}
    if use_all:
        per_record[ALL_K_SENTINEL] = []

    # micro sums
    micro_sums = {
        k: {"tp_evidence": 0.0, "retrieved": 0.0, "gold": 0.0, "rel_chunks": 0.0, "dcg": 0.0, "idcg": 0.0}
        for k in ks
    }
    if use_all:
        micro_sums[ALL_K_SENTINEL] = {"tp_evidence": 0.0, "retrieved": 0.0, "gold": 0.0,
                                      "rel_chunks": 0.0, "dcg": 0.0, "idcg": 0.0}

    iterator = range(len(dataset))
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(dataset), desc="Eval records")

    for idx in iterator:
        item = dataset[idx]
        qid = item.get("question_id", f"idx{idx}")

        # Skip record không có evidences
        if not _has_evidence(item):
            skipped_no_evidence.append((idx, qid))
            continue

        try:
            per_k_metrics, per_k_counts = eval_one_record(
                item=item,
                ks=ks,
                use_all=use_all,
                contain_threshold=contain_threshold,
                precision_mode=precision_mode,
            )
        except Exception as e:
            failed.append((idx, repr(e)))
            continue

        for k, m in per_k_metrics.items():
            per_record.setdefault(k, []).append((idx, qid, m))
            for mk in METRIC_KEYS:
                macro_sums[k][mk] += m.get(mk, 0.0)
            counts[k] = counts.get(k, 0) + 1

        for k, c in per_k_counts.items():
            ms = micro_sums.setdefault(k, {"tp_evidence": 0.0, "retrieved": 0.0, "gold": 0.0,
                                           "rel_chunks": 0.0, "dcg": 0.0, "idcg": 0.0})
            for key in ms:
                ms[key] += c[key]

    macro_avgs = {
        k: {m: (macro_sums[k][m] / counts[k]) if counts[k] > 0 else 0.0 for m in METRIC_KEYS}
        for k in counts.keys()
    }

    micro_avgs = {k: _counts_to_metrics(c, precision_mode=precision_mode) for k, c in micro_sums.items()}

    # trả thêm danh sách skip
    return macro_avgs, micro_avgs, counts, failed, per_record, micro_sums, skipped_no_evidence


# ========================= REPORT/DUMP =========================
def _k_label(k: int) -> str:
    return "ALL" if k == ALL_K_SENTINEL else str(k)


def print_report_both(macro_avgs, micro_avgs, counts, failed, skipped_no_evidence):
    print("=== Retrieval evaluation ===")
    for k in sorted(counts.keys()):
        ma = macro_avgs[k]
        mi = micro_avgs[k]
        label = _k_label(k)
        print(f"--- @ {label} (n={counts[k]}) ---")
        print(f"Macro  | P: {ma['precision']:.4f}  R: {ma['recall']:.4f}  F1: {ma['f1']:.4f}  nDCG: {ma['ndcg']:.4f}")
        print(f"Micro  | P: {mi['precision']:.4f}  R: {mi['recall']:.4f}  F1: {mi['f1']:.4f}  nDCG: {mi['ndcg']:.4f}")

    if skipped_no_evidence:
        print(f"\n[Info] Skipped {len(skipped_no_evidence)} record(s) without evidences.")

    if failed:
        print(f"[Warn] {len(failed)} record(s) lỗi, bỏ qua:")
        for i, (idx, err) in enumerate(failed[:10], start=1):
            print(f"  {i}. index={idx}, err={err}")
        if len(failed) > 10:
            print(f"  ... và {len(failed) - 10} lỗi khác")


def dump_bad_cases(
    dataset: List[Dict[str, Any]],
    per_record: dict,
    out_path: str,
    ks: Tuple[int, ...],
    thresholds: Dict[str, float] | None = None,
    bottoms: Dict[str, int] | None = None,
    include_question: bool = True,
    use_all: bool = False,
):
    thresholds = thresholds or {}
    bottoms = bottoms or {}
    by_k = {}

    all_ks = list(ks)
    if use_all:
        all_ks.append(ALL_K_SENTINEL)

    for k in all_ks:
        arr = per_record.get(k, [])
        selected = []

        for metric, thr in thresholds.items():
            for idx, qid, m in arr:
                if m.get(metric, 0.0) < float(thr):
                    selected.append(("thresh", metric, thr, idx, qid, m))

        for metric, n in bottoms.items():
            worst = sorted(arr, key=lambda t: t[2].get(metric, 0.0))[:int(n)]
            for idx, qid, m in worst:
                selected.append(("bottom", metric, n, idx, qid, m))

        seen = set()
        items = []
        for kind, metric, param, idx, qid, m in selected:
            key = (idx, qid)
            if key in seen:
                continue
            seen.add(key)
            entry = {
                "idx": idx,
                "question_id": qid,
                "reason": (
                    f"{kind}:{metric}<{param}" if kind == "thresh" else f"{kind}:{metric}@{param}"
                ),
                "metrics": {mk: float(m.get(mk, 0.0)) for mk in METRIC_KEYS},
            }
            if include_question:
                entry["question"] = dataset[idx].get("question")
            items.append(entry)

        by_k[_k_label(k)] = items

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config": {"thresholds": thresholds, "bottoms": bottoms},
        "counts": {str(_k_label(k)): len(by_k[_k_label(k)]) for k in by_k.keys()},
        "by_k": by_k,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Dumped bad cases to: {out_path}")


def dump_bad_cases_csv(json_path: str, csv_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for k, items in data["by_k"].items():
        for it in items:
            rows.append({
                "k": k,
                "idx": it["idx"],
                "question_id": it["question_id"],
                "reason": it["reason"],
                "precision": it["metrics"]["precision"],
                "recall": it["metrics"]["recall"],
                "f1": it["metrics"]["f1"],
                "ndcg": it["metrics"]["ndcg"],
                "question": it.get("question", ""),
            })

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")
    print(f"✅ CSV written: {csv_path}")


# ========================= SAVE RESULTS =========================
def save_eval_results(
    out_dir: str,
    out_filename: str,
    meta: Dict[str, Any],
    macro_avgs: Dict[int, Dict[str, float]],
    micro_avgs: Dict[int, Dict[str, float]],
    counts: Dict[int, int],
    failed: List[Tuple[int, str]],
    per_record: Dict[int, List[Tuple[int, str, Dict[str, float]]]],
    micro_sums: Dict[int, Dict[str, float]],
    skipped_no_evidence: List[Tuple[int, str]],
) -> str:
    """
    Lưu file JSON kết quả đầy đủ (meta + macro/micro + per-record + micro_sums + skipped list).
    out_filename nên là basename của file input để "giống tên file input".
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)

    def _dict_with_str_k(d: Dict[int, Any]) -> Dict[str, Any]:
        return {_k_label(k): v for k, v in d.items()}

    payload: Dict[str, Any] = {
        "meta": meta,
        "macro_avgs": _dict_with_str_k(macro_avgs),
        "micro_avgs": _dict_with_str_k(micro_avgs),
        "counts": _dict_with_str_k(counts),
        "failed": failed,
        "per_record": {
            _k_label(k): [
                {"idx": idx, "question_id": qid, "metrics": m}
                for (idx, qid, m) in arr
            ]
            for k, arr in per_record.items()
        },
        "micro_sums": _dict_with_str_k(micro_sums),  # để kiểm tra lại phép tính micro
        "skipped_no_evidence": [{"idx": i, "question_id": q} for (i, q) in skipped_no_evidence],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"👌 Saved eval results -> {out_path}")
    return out_path


# ========================= CLI =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate given chunks vs evidences (no retrieve)")
    p.add_argument("--input", nargs="+", required=True,
                   help="Path(s) to dataset json (sẽ chạy & lưu từng file tương ứng)")
    p.add_argument("--ks", default="3,5,10",
                   help="Comma list, vd: 1,3,5,10 hoặc thêm 'all' hay '0' để dùng toàn bộ chunks")
    p.add_argument("--contain-threshold", type=float, default=0.85)
    p.add_argument("--precision-mode", choices=["ir", "legacy"], default="ir",
                   help="precision 'ir' = rel_chunks/retrieved (chuẩn), 'legacy' = tp_evidence/retrieved")

    p.add_argument("--bad-json", default=None, help="Path to dump bad cases JSON")
    p.add_argument("--bad-csv", default=None, help="(Optional) Also dump CSV from bad-json")
    p.add_argument("--thr-recall", type=float, default=None, help="Chọn record có recall<thr")
    p.add_argument("--thr-f1", type=float, default=None, help="Chọn bottom theo f1<thr")
    p.add_argument("--bottom-f1", type=int, default=None, help="Chọn bottom-N theo f1")

    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help=f"Thư mục lưu kết quả (mặc định: {DEFAULT_OUT_DIR})")
    p.add_argument("--out", dest="out_file", default=None,
                   help="Đường dẫn FILE JSON output cụ thể (dùng khi --input có 1 file).")
    return p.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    ks, use_all = _parse_ks(args.ks)

    # Chạy & lưu riêng cho từng input để tên file output trùng với input
    for pth in args.input:
        dataset = load_dataset(pth)
        macro_avgs, micro_avgs, counts, failed, per_record, micro_sums, skipped_no_evidence = eval_dataset(
            dataset=dataset,
            ks=ks,
            use_all=use_all,
            contain_threshold=args.contain_threshold,
            precision_mode=args.precision_mode,
        )

        print(f"\n=== File: {pth} ===")
        print_report_both(macro_avgs, micro_avgs, counts, failed, skipped_no_evidence)

        meta = {
            "input_file": pth,
            "ks": [(_k_label(k)) for k in counts.keys()],
            "contain_threshold": args.contain_threshold,
            "precision_mode": args.precision_mode,
            "skipped_no_evidence": len(skipped_no_evidence),
        }
        if args.out_file and len(args.input) > 1:
            raise SystemExit("❌ --out chỉ dùng khi --input có đúng 1 file. Hãy dùng --out-dir cho nhiều input.")

        if args.out_file:
            out_dir = os.path.dirname(args.out_file) or "."
            out_filename = os.path.basename(args.out_file)
            if not out_filename.lower().endswith(".json"):
                out_filename += ".json"
        else:
            out_dir = args.out_dir
            out_filename = os.path.basename(pth)
            if not out_filename.lower().endswith(".json"):
                out_filename += ".json"

        save_eval_results(
            out_dir=out_dir,
            out_filename=out_filename,
            meta=meta,
            macro_avgs=macro_avgs,
            micro_avgs=micro_avgs,
            counts=counts,
            failed=failed,
            per_record=per_record,
            micro_sums=micro_sums,
            skipped_no_evidence=skipped_no_evidence,
        )

    # Bad-case dump (áp dụng cho file cuối cùng đã chạy)
    if args.bad_json:
        thresholds = {}
        if args.thr_recall is not None:
            thresholds["recall"] = float(args.thr_recall)
        if args.thr_f1 is not None:
            thresholds["f1"] = float(args.thr_f1)
        bottoms = {}
        if args.bottom_f1 is not None:
            bottoms["f1"] = int(args.bottom_f1)
        dump_bad_cases(
            dataset=dataset,
            per_record=per_record,
            out_path=args.bad_json,
            ks=ks,
            thresholds=thresholds or None,
            bottoms=bottoms or None,
            include_question=True,
            use_all=use_all,
        )
        if args.bad_csv:
            dump_bad_cases_csv(args.bad_json, args.bad_csv)


if __name__ == "__main__":
    main()

"""
python /home/hungpv/projects/custom_hippo/evaluator.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.hippo.json \
  --ks all,3,5,10 \
  --precision-mode ir \
  --contain-threshold 0.5
python /home/hungpv/projects/custom_hippo/evaluator.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.retrieved.json \
  --ks all,3,5,10 \
  --precision-mode ir \
  --contain-threshold 0.5
python /home/hungpv/projects/custom_hippo/evaluator.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.magix.json \
  --ks all,3,5,10 \
  --precision-mode ir \
  --contain-threshold 0.5
python /home/hungpv/projects/custom_hippo/evaluator.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.magix.retrieved.json \
  --ks all,3,5,10 \
  --precision-mode ir \
  --contain-threshold 0.5

python /home/hungpv/projects/custom_hippo/evaluator.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.magix.retrieved.json \
  --ks all,3,5,10 \
  --precision-mode ir \
  --contain-threshold 0.5

"""
