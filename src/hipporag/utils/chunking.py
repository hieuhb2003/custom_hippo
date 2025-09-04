from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
import logging
from .misc_utils import compute_mdhash_id

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

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the DocumentChunker.

        Args:
            chunk_size (int): Maximum size of each chunk in characters
            chunk_overlap (int): Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Dictionary to map chunk IDs to document IDs
        self.chunk_to_doc = {}

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

        # Handle content shorter than chunk_size
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
