from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class BGEM3EmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        global_config: Optional[BaseConfig] = None,
        embedding_model_name: Optional[str] = "BAAI/bge-m3",
    ) -> None:
        super().__init__(global_config=global_config)
        self.embedding_model_name = embedding_model_name
        self._init_embedding_config()

        # Load SentenceTransformer model
        self.model = SentenceTransformer(
            self.embedding_model_name, device=None, trust_remote_code=True
        )

        # Set to evaluation mode and move to CUDA
        self.model.eval()

        # Embedding dimension is typically 1024 for BGE-M3
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "model_name_or_path": self.embedding_model_name,
                "device": None,
                "trust_remote_code": True,
            },
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                "normalize_embeddings": self.global_config.embedding_return_as_normalized,
                "convert_to_tensor": True,
                "show_progress_bar": False,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"EmbeddingConfig: {self.embedding_config}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Produce dense embeddings for a list of texts using SentenceTransformer."""
        # Encode texts using SentenceTransformer
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=self.embedding_config.encode_params["batch_size"],
            normalize_embeddings=self.embedding_config.encode_params[
                "normalize_embeddings"
            ],
            convert_to_tensor=True,
            show_progress_bar=False,
            device=None,
        )

        # Ensure embeddings are on the correct device
        # if not embeddings.is_cuda:
        #     embeddings = embeddings.to("cuda:0")

        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 12)  # Default to 12 for consistency
        normalize_embeddings = params.pop(
            "normalize_embeddings", self.embedding_config.norm
        )
        logger.debug(
            f"BGEM3Model batch_encode with batch_size: {batch_size}, normalize: {normalize_embeddings}"
        )

        # Encode in batches using SentenceTransformer's built-in batching
        if len(texts) <= batch_size:
            embs = self.encode(texts)
        else:
            # For large datasets, we can still use manual batching with progress bar
            pbar = tqdm(total=len(texts), desc="BGE-M3 Batch Encoding")
            embs_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embs = self.model.encode(
                    sentences=batch_texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=None,
                )
                embs_list.append(batch_embs)
                pbar.update(len(batch_texts))
            pbar.close()
            embs = torch.cat(embs_list, dim=0)

        # Convert to numpy and move to CPU
        embs = embs.cpu().numpy()
        return embs
