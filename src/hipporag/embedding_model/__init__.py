# from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel

# from .GritLM import GritLMEmbeddingModel
# from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel

# from .Cohere import CohereEmbeddingModel
from .BGEM3 import BGEM3EmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    # if "GritLM" in embedding_model_name:
    #     return GritLMEmbeddingModel
    # elif "NV-Embed-v2" in embedding_model_name:
    #     return NVEmbedV2EmbeddingModel
    if "bge" in embedding_model_name.lower() and "m3" in embedding_model_name.lower():
        # Match typical identifiers like BAAI/bge-m3
        return BGEM3EmbeddingModel
    # elif "contriever" in embedding_model_name:
    #     return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    # elif "cohere" in embedding_model_name:
    #     return CohereEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
