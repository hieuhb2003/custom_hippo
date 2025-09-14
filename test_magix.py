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
    parser = argparse.ArgumentParser(description="Build retrieved dataset using MAGIX-style retrieval")
    parser.add_argument("--input", default="/Users/khangtuan/Documents/memory/custom_hippo/longmemeval_0_500_v3.json")
    parser.add_argument("--output", default="outputs/retrieved_datasets/longmemeval_0_500_v3.magix.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--embedding", default="text-embedding-3-small")
    parser.add_argument("--save_dir", default="outputs/magix_build")
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--method", choices=["max", "max_top_k", "new_function", "weighted"], default="weighted")
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)
        # if args.max_docs is not None:
        #     dataset = dataset[: max(0, int(args.max_docs))]

    # Reuse HippoRAG for indexing embeddings; retrieval will be handled by magix_retrieval
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
    )

    hippo = HippoRAG(global_config=cfg)
    # Only build chunk embeddings to avoid OpenIE/LLM calls
    hippo.chunk_embedding_store.insert_strings(unique_docs)

    # Build dataset via MAGIX retrieval
    new_dataset: List[Dict[str, Any]] = []

    # Import MAGIX retrieval helpers
    from magix_retrieval import enhanced_chunk_retrieval_direct

    # Prepare lightweight storages from hippo stores
    # Map chunk_id to content
    chunk_id_to_doc_id = {cid: cid for cid in hippo.chunk_embedding_store.get_all_ids()}
    doc_id_to_doc_content = {cid: hippo.chunk_embedding_store.get_row(cid)["content"] for cid in hippo.chunk_embedding_store.get_all_ids()}
    doc_content_to_doc_idx = {v: k for k, v in doc_id_to_doc_content.items()}

    # Dummy VDB adapters by reusing HippoRAG embeddings in memory is out-of-scope
    # Here, we fallback to DPR-only via MAGIX combiner (using chunk embeddings); entities/edges empty

    class _DummyDB:
        def query(self, *args, **kwargs):
            return []

    class _ChunksVDB:
        def query(self, query_embedding, top_k, filter_lambda=None):
            # Brute-force over stored chunk embeddings
            import numpy as np
            ids = hippo.chunk_embedding_store.get_all_ids()
            if not ids:
                return []
            embs = hippo.chunk_embedding_store.get_embeddings(ids)
            sims = embs @ query_embedding.reshape(-1, 1)
            sims = np.squeeze(sims)
            order = np.argsort(sims)[::-1]
            results = []
            for idx in order[:top_k]:
                cid = ids[int(idx)]
                if filter_lambda and not filter_lambda({"__id__": cid}):
                    continue
                results.append({"__id__": cid, "score": float(sims[int(idx)])})
            return results

    entities_vdb = _DummyDB()
    relationships_vdb = _DummyDB()
    text_chunks_db = {cid: {"content": hippo.chunk_embedding_store.get_row(cid)["content"]} for cid in hippo.chunk_embedding_store.get_all_ids()}
    chunks_vdb = _ChunksVDB()

    # Need a model encoder compatible with MAGIX helper: use hippo.embedding_model
    model = hippo.embedding_model

    for item in dataset:
        q = item.get("question", "")
        if not isinstance(q, str) or q.strip() == "":
            new_item = dict(item)
            new_item["chunks"] = _coerce_chunks(item)
            new_dataset.append(new_item)
            continue

        try:
            corpus_ids = enhanced_chunk_retrieval_direct(
                query=q,
                full_graph=None,
                entities_vdb=entities_vdb,
                relationships_vdb=relationships_vdb,
                text_chunks_db=text_chunks_db,
                chunks_vdb=chunks_vdb,
                model=model,
                chunk_id_to_doc_id=chunk_id_to_doc_id,
                doc_id_to_doc_content=doc_id_to_doc_content,
                doc_content_to_doc_idx=doc_content_to_doc_idx,
                top_k_entities=0,
                top_k_edges=0,
                top_k_chunks=max(1, int(args.top_k)),
                a=args.a,
                b=args.b,
                c=args.c,
                relevance_threshold=args.relevance_threshold,
                method=(None if args.method == "weighted" else args.method),
            )
            retrieved_chunks = [doc_id_to_doc_content[cid] for cid in corpus_ids]
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
python /home/hungpv/projects/custom_hippo/test_magix.py \
  --input /home/hungpv/projects/custom_hippo/outputs/retrieved_datasets/longmemeval_0_500_v3.magix.json \
  --output outputs/retrieved_datasets/longmemeval_0_500_v3.magix.json \
  --top_k 10 \
  --llm "Qwen/Qwen3-8B" \
  --embedding "BAAI/bge-m3" \
  --save_dir outputs/magix_build \
  --a 0.3 --b 0.5 --c 0.5 \
  --relevance_threshold 0.7 \
  --method weighted
"""