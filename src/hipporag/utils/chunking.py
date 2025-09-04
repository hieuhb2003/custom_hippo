from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import logging
from .misc_utils import compute_mdhash_id
import json

try:
    import tiktoken
except Exception:
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    Class representing a chunk of a document.
    """

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int


class DocumentChunker:
    """
    Class for chunking documents into smaller pieces.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        token_size: int = None,
        token_overlap: int = None,
        encoding_name: str = "o200k_base",
    ):
        """
        Initialize the DocumentChunker.

        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_size = token_size
        self.token_overlap = token_overlap
        self.encoding_name = encoding_name
        self.encoding = None
        if token_size is not None and tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception:
                logger.warning(
                    f"tiktoken encoding '{encoding_name}' not found. Falling back to char-based chunking."
                )
                self.encoding = None
        # Dictionary to map chunk IDs to document IDs
        self.chunk_to_doc = {}

    def _chunk_document_by_tokens(
        self, doc_id: str, content: str
    ) -> List[DocumentChunk]:
        chunks = []
        tokens = self.encoding.encode(content)
        if len(tokens) == 0:
            return chunks
        start = 0
        idx = 0
        size = int(self.token_size or 800)
        overlap = int(self.token_overlap or 200)
        while start < len(tokens):
            end = min(start + size, len(tokens))
            piece = self.encoding.decode(tokens[start:end])
            chunk_id = compute_mdhash_id(piece, prefix="chunk-")
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id, doc_id=doc_id, content=piece, chunk_index=idx
                )
            )
            self.chunk_to_doc[chunk_id] = doc_id
            idx += 1
            if end >= len(tokens):
                break
            start = max(0, end - overlap)
        return chunks

    def chunk_document(self, doc_id: str, content: str) -> List[DocumentChunk]:
        """
        Split a document into overlapping chunks.

        Args:
            doc_id (str): Identifier for the document
            content (str): Content of the document to chunk

        Returns:
            List[DocumentChunk]: List of document chunks with IDs and metadata
        """
        chunks = []

        # Check if content is empty
        if not content.strip():
            logger.warning(f"Empty document content for doc_id: {doc_id}")
            return chunks

        # Token-based chunking if configured and available
        if self.encoding is not None and self.token_size is not None:
            return self._chunk_document_by_tokens(doc_id, content)

        # Fallback: Handle content shorter than chunk_size (char-based)
        if len(content) <= self.chunk_size:
            chunk_id = compute_mdhash_id(content, prefix="chunk-")
            chunk = DocumentChunk(
                chunk_id=chunk_id, doc_id=doc_id, content=content, chunk_index=0
            )
            chunks.append(chunk)
            self.chunk_to_doc[chunk_id] = doc_id
            return chunks

        # Split by paragraphs first to avoid breaking in the middle of a sentence
        paragraphs = re.split(r"\n\s*\n", content)
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()

            if not para:
                continue

            # If adding paragraph would exceed chunk size, create a new chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk_id = compute_mdhash_id(current_chunk, prefix="chunk-")
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=current_chunk,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                self.chunk_to_doc[chunk_id] = doc_id
                chunk_index += 1

                # Start new chunk with overlap
                last_words = " ".join(
                    current_chunk.split()[-self.chunk_overlap // 10 :]
                )
                current_chunk = last_words + "\n\n" + para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_id = compute_mdhash_id(current_chunk, prefix="chunk-")
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=current_chunk,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)
            self.chunk_to_doc[chunk_id] = doc_id

        return chunks

    def chunk_documents(self, docs: Dict[str, str]) -> List[DocumentChunk]:
        """
        Process multiple documents and split them into chunks.

        Args:
            docs (Dict[str, str]): Dictionary mapping document IDs to content

        Returns:
            List[DocumentChunk]: List of all document chunks
        """
        all_chunks = []

        for doc_id, content in docs.items():
            doc_chunks = self.chunk_document(doc_id, content)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def get_chunk_to_doc_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from chunk IDs to document IDs.

        Returns:
            Dict[str, str]: Dictionary mapping chunk IDs to document IDs
        """
        return self.chunk_to_doc.copy()

    def get_doc_to_chunks_mapping(self) -> Dict[str, List[str]]:
        doc_to_chunks: Dict[str, List[str]] = {}
        for c_id, d_id in self.chunk_to_doc.items():
            doc_to_chunks.setdefault(d_id, []).append(c_id)
        return doc_to_chunks

    def dump_mappings(self, path: str) -> None:
        mappings = {
            "chunk_to_doc": self.get_chunk_to_doc_mapping(),
            "doc_to_chunks": self.get_doc_to_chunks_mapping(),
        }
        with open(path, "w") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
