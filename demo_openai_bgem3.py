import os
from src.hipporag import HippoRAG
from multiprocessing import freeze_support


def main():
    # Sample docs
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County.",
    ]

    # Configure OpenAI LLM and BGE-M3 embedding
    save_dir = "outputs/openai_bgem3_2"
    llm_model_name = "gpt-4o-mini"
    embedding_model_name = "BAAI/bge-m3"  # will be routed to BGEM3EmbeddingModel

    # HippoRAG instance
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
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
