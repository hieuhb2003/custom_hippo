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

    parser = argparse.ArgumentParser(description="HippoRAG + Magix retrieval: dataset-based evaluation")
    parser.add_argument("--input", type=str, default=None, help="Path to dataset JSON file. If omitted, uses a small built-in toy set.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results JSON.")
    parser.add_argument("--llm", default="Qwen/Qwen3-8B")
    parser.add_argument("--embedding", default="BAAI/bge-m3")
    parser.add_argument("--save_dir", default="outputs/local_test_magix")
    parser.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--llm_api_key", default="EMPTY")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--dump_retrieved", type=str, default=None, help="If set, dump a new dataset with retrieved chunks to this JSON path")
    parser.add_argument("--dump_top_k", type=int, default=10, help="Top-K chunks per question to dump when --dump_retrieved is set")

    # Magix knobs
    parser.add_argument("--top_k_entities", type=int, default=100)
    parser.add_argument("--top_k_edges", type=int, default=100)
    parser.add_argument("--top_k_chunks", type=int, default=100)
    parser.add_argument("--a", type=float, default=0.3)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--method", type=str, default=None, choices=[None, "max", "max_top_k", "new_function"], help="Magix fusion method")
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
            dataset = dataset

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

    magix_kwargs = dict(
        top_k_entities=max(0, int(args.top_k_entities)),
        top_k_edges=max(0, int(args.top_k_edges)),
        top_k_chunks=max(1, int(args.top_k_chunks)),
        a=float(args.a),
        b=float(args.b),
        c=float(args.c),
        relevance_threshold=float(args.relevance_threshold),
        method=args.method,
    )

    try: 
        retrieved = hipporag.retrieve_magix(
            queries=queries,
            num_to_retrieve=max(1, int(args.dump_top_k)),
            **magix_kwargs,
        )
    except Exception as e:
        print(f"[error] retrieve_magix failed: {e}")
        return

    if args.input is not None:
        new_dataset: List[Dict[str, Any]] = []
        for item, qs in zip(dataset, retrieved):
            new_item = dict(item)
            new_item["chunks"] = list(qs.docs)[: args.dump_top_k]
            new_dataset.append(new_item)

        out_path = args.output or args.dump_retrieved
        if not out_path:
            out_path = "outputs/retrieved_datasets/magix.retrieved.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved retrieved dataset to {out_path}")
    else:
        # Toy mode: print a compact preview
        for q, qs in zip(queries, retrieved):
            print("Q:", q)
            print("Top chunks:")
            for d in qs.docs[: args.dump_top_k]:
                print(" -", d[:200].replace("\n", " ") + ("..." if len(d) > 200 else ""))
            print()


if __name__ == "__main__":
    main()
"""
python /home/hungpv/projects/custom_hippo/tests_magix.py \
  --input /home/hungpv/projects/custom_hippo/longmemeval_0_500_v3.json \
  --output outputs/retrieved_datasets/longmemeval_0_500_v3.magix.retrieved.json \
  --llm "Qwen/Qwen3-8B" --embedding "BAAI/bge-m3" \
  --save_dir outputs/local_test_magix_fresh6 \
  --llm_base_url http://localhost:8000/v1 --llm_api_key EMPTY \
  --dump_top_k 10 \
  --top_k_entities 200 --top_k_edges 200 --top_k_chunks 50 \
  --a 0.3 --b 0.5 --c 0.5 --relevance_threshold 0.3
"""