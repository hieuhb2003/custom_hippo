import os
from typing import List

from src.hipporag.llm.openai_gpt import CacheOpenAI
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.llm_utils import TextChatMessage


def main():
    config = BaseConfig(
        llm_name="Qwen/Qwen3-8B",  
        llm_base_url="http://localhost:8000/v1",
        save_dir="outputs/llm_ping",
    )
    config.llm_api_key = "EMPTY"
    config.llm_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    llm = CacheOpenAI.from_experiment_config(config)

    messages: List[TextChatMessage] = [
        {"role": "user", "content": "hello world"},
    ]

    response, metadata, cache_hit = llm.infer(messages)

    print("=== LLM Response ===")
    print(response)
    print("=== Metadata ===")
    print(metadata)
    print("cache_hit:", cache_hit)


if __name__ == "__main__":
    main()


"""
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    { "role": "user", "content": "tell me something fun" }
  ],
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}'
"""