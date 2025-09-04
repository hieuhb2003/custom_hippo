import os
from src.hipporag import HippoRAG
from multiprocessing import freeze_support
from src.hipporag.utils.config_utils import BaseConfig

cfg = BaseConfig()
cfg.enable_chunking = True
cfg.chunk_tokens = 800
cfg.chunk_overlap_tokens = 200
cfg.chunk_encoding = "o200k_base"

cfg.enable_api_key_rotation = True
cfg.llm_base_url = "https://openrouter.ai/api/v1"
cfg.api_key_file_path = "/Users/hieunguyenmanh/hippo2/HippoRAG/outputs/key.txt"  # mỗi dòng 1 key
cfg.api_key_daily_quota = 20


def main():
    # Sample docs
    docs = [
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.",
    ]

    # Configure OpenAI LLM and BGE-M3 embedding
    save_dir = "outputs/openai_bgem3_multi_key"
    llm_model_name = "openai/gpt-oss-20b:free"
    embedding_model_name = "BAAI/bge-m3"  # will be routed to BGEM3EmbeddingModel

    # HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        global_config=cfg,
    )

    # Index
    hipporag.index(docs=docs)

    # Queries
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?",
    ]

    # Gold for evaluation
    answers = [["Politician"], ["By going to the ball."], ["Rockland County"]]
    gold_docs = [
        ["George Rankin is a politician."],
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        ],
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County.",
        ],
    ]

    print(
        hipporag.rag_qa(
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=answers,
        )
    )


if __name__ == "__main__":
    freeze_support()
    main()
