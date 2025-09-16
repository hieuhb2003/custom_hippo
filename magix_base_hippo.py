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
    parser = argparse.ArgumentParser(
        description="Build full Hippo index (OpenIE+graph) then retrieve via MAGIX"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_dir", default="outputs/magix_base_hippo")
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--llm_base_url", default=None, help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1 for vLLM)")
    parser.add_argument("--llm_api_key", default=None, help="API key for OpenAI-compatible server (use 'sk-' for local vLLM if needed)")
    parser.add_argument("--embedding", default="text-embedding-3-small")
    parser.add_argument("--openie_mode", choices=["online", "offline"], default="offline", help="Use offline (vLLM in-process batch) to speed up indexing")
    # Offline vLLM tuning flags
    parser.add_argument("--llm_num_gpus", type=int, default=None, help="Number of GPUs for vLLM offline (default: auto)")
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=None, help="GPU memory utilization ratio for vLLM offline (e.g., 0.2)")
    parser.add_argument("--llm_max_model_len", type=int, default=None, help="Max model length for vLLM offline (e.g., 2048)")
    parser.add_argument("--llm_quantization", default=None, help="Quantization mode for vLLM offline (e.g., 'bitsandbytes', 'awq')")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_k_entities", type=int, default=50)
    parser.add_argument("--top_k_edges", type=int, default=0)
    parser.add_argument("--method", choices=["max", "max_top_k", "new_function", "weighted"], default="weighted")
    parser.add_argument("--a", type=float, default=0.6)
    parser.add_argument("--b", type=float, default=0.3)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--skip_index", action="store_true", help="Use existing index in save_dir")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument(
        "--index_scope",
        choices=["all", "selected"],
        default="all",
        help="When indexing, use all docs (default) or only chunks from selected queries",
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Limit number of questions to retrieve (for quick eval)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    # Build the document corpus from dataset chunks
    all_docs: List[str] = []
    for item in dataset:
        all_docs.extend(_coerce_chunks(item))

    if args.max_docs is not None:
        all_docs = all_docs[: max(0, int(args.max_docs))]

    # Deduplicate
    seen = set()
    unique_docs: List[str] = []
    for d in all_docs:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)

    cfg = BaseConfig(
        save_dir=args.save_dir,
        llm_name=args.llm,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        embedding_model_name=args.embedding,
        openie_mode=args.openie_mode,
    )

    # Pass optional offline tuning knobs to config for vLLM offline backend
    if args.llm_num_gpus is not None:
        setattr(cfg, "llm_num_gpus", int(args.llm_num_gpus))
    if args.llm_gpu_memory_utilization is not None:
        setattr(cfg, "llm_gpu_memory_utilization", float(args.llm_gpu_memory_utilization))
    if args.llm_max_model_len is not None:
        setattr(cfg, "llm_max_model_len", int(args.llm_max_model_len))
    if args.llm_quantization is not None:
        setattr(cfg, "llm_quantization", args.llm_quantization)

    hippo = HippoRAG(global_config=cfg)

    # Prepare queries
    queries: List[str] = []
    keep_indices: List[int] = []
    for idx, item in enumerate(dataset):
        q = item.get("question", "")
        if isinstance(q, str) and q.strip():
            queries.append(q)
            keep_indices.append(idx)

    # Limit number of queries for quick test
    if args.max_queries is not None and len(queries) > args.max_queries:
        queries = queries[: args.max_queries]
        keep_indices = keep_indices[: args.max_queries]

    # Decide indexing corpus scope
    if args.index_scope == "selected" and keep_indices:
        docs_for_index: List[str] = []
        for idx in keep_indices:
            docs_for_index.extend(_coerce_chunks(dataset[idx]))
        # Deduplicate
        seen_idx = set()
        selected_unique_docs: List[str] = []
        for d in docs_for_index:
            if d not in seen_idx:
                seen_idx.add(d)
                selected_unique_docs.append(d)
        index_docs = selected_unique_docs
    else:
        index_docs = unique_docs

    # Build full index only when requested (default)
    if not args.skip_index:
        # This runs OpenIE, encodes entities/facts, constructs graph, and persists under save_dir
        hippo.index(index_docs)

    # Retrieve via MAGIX using Hippo's real stores
    if queries:
        results = hippo.retrieve_magix(
            queries=queries,
            num_to_retrieve=max(1, int(args.top_k)),
            top_k_entities=max(0, int(args.top_k_entities)),
            top_k_edges=max(0, int(args.top_k_edges)),
            method=(None if args.method == "weighted" else args.method),
            a=args.a,
            b=args.b,
            c=args.c,
            relevance_threshold=args.relevance_threshold,
        )
    else:
        results = []

    # Assemble new dataset
    new_dataset: List[Dict[str, Any]] = []
    res_ptr = 0
    for idx, item in enumerate(dataset):
        new_item = dict(item)
        q = item.get("question", "")
        if isinstance(q, str) and q.strip() and idx in keep_indices:
            sol = results[res_ptr]
            res_ptr += 1
            new_item["chunks"] = list(sol.docs)
        else:
            new_item["chunks"] = _coerce_chunks(item)
        new_dataset.append(new_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved MAGIX-based dataset to {args.output}")
    print("You can now evaluate it, e.g.:")
    print(
        f"python evaluator.py --input {args.output} --ks 3,5,10 --precision-mode ir --contain-threshold 0.85"
    )


if __name__ == "__main__":
    main()


