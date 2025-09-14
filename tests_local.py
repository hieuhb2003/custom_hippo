import os
from typing import List, Any, Dict
import json
import argparse
import logging

from src.hipporag import HippoRAG


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

    parser = argparse.ArgumentParser(description="HippoRAG local test: dataset-based evaluation")
    parser.add_argument("--input", type=str, default=None, help="Path to dataset JSON file. If omitted, uses a small built-in toy set.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results JSON.")
    parser.add_argument("--llm", default="Qwen/Qwen3-8B")
    parser.add_argument("--embedding", default="BAAI/bge-m3")
    parser.add_argument("--save_dir", default="outputs/local_test")
    parser.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--llm_api_key", default="EMPTY")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--dump_retrieved", type=str, default=None, help="If set, dump a new dataset with retrieved chunks to this JSON path")
    parser.add_argument("--dump_top_k", type=int, default=10, help="Top-K chunks per question to dump when --dump_retrieved is set")
    args = parser.parse_args()

    llm_extra_body = {"chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}}

    if args.input is None:
        docs = [
            "Oliver Badman is a politician.",
            "George Rankin is a politician.",
            "Thomas Marwick is a politician.",
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince.",
            "Erik Hort's birthplace is Montebello.",
            "Marina is bom in Minsk.",
            "Montebello is a part of Rockland County."
        ]
        queries = [
            "What is George Rankin's occupation?",
            "How did Cinderella reach her happy ending?",
            "What county is Erik Hort's birthplace a part of?"
        ]
        answers = [["Politician"], ["By going to the ball."], ["Rockland County"]]
        gold_docs = [
            ["George Rankin is a politician."],
            [
                "Cinderella attended the royal ball.",
                "The prince used the lost glass slipper to search the kingdom.",
                "When the slipper fit perfectly, Cinderella was reunited with the prince.",
            ],
            ["Erik Hort's birthplace is Montebello.", "Montebello is a part of Rockland County."],
        ]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            dataset: List[Dict[str, Any]] = json.load(f)
            # dataset = dataset[:10]

        all_docs: List[str] = []
        for item in dataset:
            chunks = _coerce_chunks(item)
            if chunks:
                all_docs.extend(chunks)
            else:
                for k in ("doc", "text", "content"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        all_docs.append(v)
                        break

        seen = set()
        docs = []
        for d in all_docs:
            if d not in seen:
                seen.add(d)
                docs.append(d)

        # Queries
        queries = []
        for item in dataset:
            q = item.get("question")
            if isinstance(q, str) and q.strip():
                queries.append(q)

        gold_docs = item.get("gold_docs_list") if dataset and isinstance(dataset[0].get("gold_docs_list"), list) else None
        gold_answers = item.get("gold_answers_list") if dataset and isinstance(dataset[0].get("gold_answers_list"), list) else None
        if gold_docs is None and all(isinstance(x.get("gold_docs"), list) for x in dataset if isinstance(x, dict)):
            gold_docs = [x.get("gold_docs") for x in dataset]
        if gold_answers is None and all(isinstance(x.get("answers"), list) for x in dataset if isinstance(x, dict)):
            gold_answers = [x.get("answers") for x in dataset]
        answers = gold_answers

    hipporag = HippoRAG(
        save_dir=args.save_dir,
        llm_model_name=args.llm,
        embedding_model_name=args.embedding,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_extra_body=llm_extra_body,
    )

    hipporag.index(docs=docs)

    if "gold_docs" in locals() and gold_docs is not None and "answers" in locals() and answers is not None:
        result = hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)
    else:
        result = hipporag.rag_qa(queries=queries)

    # Optional: dump retrieved chunks as a new dataset for evaluator.py
    if args.dump_retrieved and args.input is not None:
        try:
            retrieved = hipporag.retrieve(queries=queries, num_to_retrieve=max(1, int(args.dump_top_k)))
            # Build new dataset keeping original fields, replacing chunks with retrieved ones
            new_dataset: List[Dict[str, Any]] = []
            for item, qs in zip(dataset, retrieved):
                new_item = dict(item)
                new_item["chunks"] = list(qs.docs)[: args.dump_top_k]
                new_dataset.append(new_item)
            os.makedirs(os.path.dirname(args.dump_retrieved) or ".", exist_ok=True)
            with open(args.dump_retrieved, "w", encoding="utf-8") as f:
                json.dump(new_dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved retrieved dataset to {args.dump_retrieved}")
        except Exception as e:
            print(f"[warn] Failed to dump retrieved dataset: {e}")

    if isinstance(result, tuple) and len(result) >= 3:
        qs, msgs, metas = result[:3]
    else:
        qs, msgs, metas = result

    print((result[-2:]) if isinstance(result, tuple) and len(result) >= 5 else (len(qs), "responses", len(msgs)))

    if args.output:
        out_obj = {
            "num_queries": len(qs),
            "responses": msgs[:5],
            "meta_sample": metas[:3],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
"""
python /home/hungpv/projects/custom_hippo/tests_local.py \
  --input /home/hungpv/projects/custom_hippo/longmemeval_0_500_v3.json \
  --output outputs/result.json \
  --llm "Qwen/Qwen3-8B" \
  --embedding "BAAI/bge-m3" \
  --save_dir outputs/local_test \
  --llm_base_url http://localhost:8000/v1 \
  --llm_api_key EMPTY \
  --dump_retrieved outputs/retrieved_datasets/longmemeval_0_500_v3.retrieved.json \
  --dump_top_k 10
"""