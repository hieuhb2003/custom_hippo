import argparse
import json
import os
from typing import Any, Dict, List

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig


def _coerce_chunks(item: Dict[str, Any]) -> List[str]:
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


def main():
    parser = argparse.ArgumentParser(description="Build retrieved dataset using HippoRAG (subgraph-PPR version)")
    parser.add_argument("--input", default="/Users/khangtuan/Documents/memory/custom_hippo/longmemeval_0_500_v3.json")
    parser.add_argument("--output", default="outputs/retrieved_datasets/longmemeval_0_500_v3.retrieved.json")
    parser.add_argument("--top_k", type=int, default=10, help="Number of retrieved chunks per question to keep")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--embedding", default="text-embedding-3-small")
    parser.add_argument("--save_dir", default="outputs/ppr_build")
    parser.add_argument("--subgraph_depth", type=int, default=1)
    parser.add_argument("--subgraph_top_k_entities", type=int, default=10)
    parser.add_argument("--subgraph_top_k_chunks_per_entity", type=int, default=10)
    parser.add_argument("--subgraph_weight_method", choices=["local", "global"], default="local")
    parser.add_argument("--subgraph_alpha", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)
        dataset = dataset[:10]

    all_docs: List[str] = []
    for item in dataset:
        all_docs.extend(_coerce_chunks(item))

    seen = set()
    unique_docs = []
    for d in all_docs:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)

    cfg = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm,
        embedding_model_name=args.embedding,
        enable_subgraph_ppr=True,
        subgraph_top_k_entities=args.subgraph_top_k_entities,
        subgraph_depth=args.subgraph_depth,
        subgraph_top_k_chunks_per_entity=args.subgraph_top_k_chunks_per_entity,
        subgraph_weight_method=args.subgraph_weight_method,
        subgraph_alpha=args.subgraph_alpha,
    )

    hippo = HippoRAG(global_config=cfg)
    hippo.index(unique_docs)

    new_dataset: List[Dict[str, Any]] = []
    for item in dataset:
        q = item.get("question", "")
        if not isinstance(q, str) or q.strip() == "":
            new_item = dict(item)
            new_item["chunks"] = _coerce_chunks(item)
            new_dataset.append(new_item)
            continue

        try:
            res = hippo.retrieve([q], num_to_retrieve=max(1, int(args.top_k)))
            qs = res[0]
            retrieved_chunks = list(qs.docs)[: args.top_k]
        except Exception:
            retrieved_chunks = _coerce_chunks(item)

        new_item = dict(item)
        new_item["chunks"] = retrieved_chunks
        new_dataset.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved retrieved dataset to {args.output}")
    print("You can now evaluate it, e.g.:")
    print(
        f"python evaluator.py --input {args.output} --ks 3,5,10 --precision-mode ir --contain-threshold 0.85"
    )


if __name__ == "__main__":
    main()

"""
python test_PPR.py \
    --input /Users/khangtuan/Documents/memory/custom_hippo/longmemeval_0_500_v3.json \
    --output outputs/retrieved_datasets/longmemeval_0_500_v3.retrieved.json \
    --top_k 10 \
    --llm gpt-4o-mini \
    --embedding text-embedding-3-small \
    --save_dir outputs/ppr_build \
    --subgraph_depth 1 \
    --subgraph_top_k_entities 10 \
    --subgraph_top_k_chunks_per_entity 10 \
    --subgraph_weight_method local \
    --subgraph_alpha 0.5 \
"""