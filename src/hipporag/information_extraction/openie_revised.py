import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, TypedDict

from tqdm import tqdm

from ..llm import BaseLLM
from ..prompts import PromptTemplateManager
from ..utils.misc_utils import TripleRawOutput
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from .entity_description import EntityDescriptionGenerator, EntityDescriptionOutput

logger = logging.getLogger(__name__)


class RevisedOpenIE:
    """
    Implementation of OpenIE (Open Information Extraction) using revised approach:
    1. Direct triplet extraction from text
    2. Entity description generation
    3. Simple entity-chunk relationships
    """

    def __init__(self, llm_model: BaseLLM):
        """
        Initialize OpenIE with a language model.

        Args:
            llm_model (BaseLLM): The language model to use for extraction
        """
        self.llm_model = llm_model
        # Initialize prompt manager (aligned with other modules)
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.entity_desc_generator = EntityDescriptionGenerator(llm_model)

    def extract_triplets(self, chunk_id: str, passage: str) -> TripleRawOutput:
        """
        Extract triplets from text using direct triplet extraction.

        Args:
            chunk_id (str): ID of the chunk
            passage (str): Text content to extract triplets from

        Returns:
            TripleRawOutput: Extracted triplets
        """
        messages = self.prompt_template_manager.render(
            name="direct_triple_extraction", passage=passage
        )
        raw_response, metadata, cache_hit = self.llm_model.infer(messages=messages)

        # Parse triplets from response
        try:
            import re, ast

            def _extract_triples_from_response(real_response: str):
                pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
                match = re.search(pattern, real_response, re.DOTALL)
                if match is None:
                    # Fallback: try to find any list-of-lists looking like triples
                    bracket_block = re.findall(r"\[[\s\S]*?\]", real_response)
                    for block in bracket_block:
                        try:
                            val = ast.literal_eval(block)
                            if isinstance(val, list) and all(
                                isinstance(x, (list, tuple)) and len(x) >= 3
                                for x in val
                            ):
                                return val
                        except Exception:
                            continue
                    return []
                return eval(match.group())["triples"]

            # Repair broken JSON if needed
            real_response = (
                fix_broken_generated_json(raw_response)
                if metadata.get("finish_reason") == "length"
                else raw_response
            )
            triplets = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triplets)
        except Exception as e:
            logger.warning(f"Error parsing triplets: {e}")
            triplets = []

        return TripleRawOutput(
            chunk_id=chunk_id,
            response=raw_response,
            triples=triplets,
            metadata=metadata,
        )

    def generate_entity_descriptions(
        self, chunk_id: str, triplets: List[List], passage: str
    ) -> EntityDescriptionOutput:
        """
        Generate descriptions for entities found in triplets.

        Args:
            chunk_id (str): ID of the chunk
            triplets (List[List]): Extracted triplets
            passage (str): Original text content

        Returns:
            EntityDescriptionOutput: Entity descriptions with metadata
        """
        # Extract unique entities from triplets
        entities = set()
        for triplet in triplets:
            if len(triplet) >= 3:
                if triplet[0]:  # Subject
                    entities.add(triplet[0])
                if triplet[2]:  # Object
                    entities.add(triplet[2])

        # Generate descriptions for the entities in ONE call using passage + triples context
        entity_descriptions, metadata = (
            self.entity_desc_generator.batch_generate_descriptions_from_triplets(
                entities=entities, triplets=triplets, passage=passage
            )
        )

        return EntityDescriptionOutput(
            chunk_id=chunk_id,
            entity_descriptions=entity_descriptions,
            metadata=metadata,
        )

    def openie(self, chunk_id: str, passage: str) -> Dict[str, Any]:
        """
        Perform the complete OpenIE process on a single passage.

        Args:
            chunk_id (str): ID of the chunk
            passage (str): Text content to process

        Returns:
            Dict[str, Any]: Dictionary with triplet and entity description results
        """
        # Extract triplets first
        triplets_output = self.extract_triplets(chunk_id=chunk_id, passage=passage)

        # Generate entity descriptions based on triplets
        entity_desc_output = self.generate_entity_descriptions(
            chunk_id=chunk_id, triplets=triplets_output.triples, passage=passage
        )

        # Generate entity+description pairs for embedding
        entity_desc_pairs = []

        for entity, description in entity_desc_output.entity_descriptions.items():
            # Combine entity and description for embedding
            entity_with_desc = f"{entity}: {description}"
            entity_desc_pairs.append(entity_with_desc)

        # Store entity+description pairs for later embedding
        entity_desc_output.entity_desc_pairs = entity_desc_pairs

        # Use original triplets only - no need to add entity-chunk triplets
        # as these relationships are maintained in the graph structure
        all_triplets = triplets_output.triples

        # Update triplets_output with combined triplets
        updated_triplets_output = TripleRawOutput(
            chunk_id=triplets_output.chunk_id,
            response=triplets_output.response,
            triples=all_triplets,
            metadata=triplets_output.metadata,
        )

        return {
            "triplets": updated_triplets_output,
            "entity_descriptions": entity_desc_output,
        }

    class ChunkInfo(TypedDict):
        num_tokens: int
        content: str
        chunk_order: List[Tuple]
        full_doc_ids: List[str]

    def batch_openie(
        self, chunks: Dict[str, "ChunkInfo"]
    ) -> Tuple[Dict[str, TripleRawOutput], List[str]]:
        """
        Process multiple chunks in parallel.

        Args:
            chunks (Dict[str, ChunkInfo]): Dictionary of chunks to process

        Returns:
            Tuple[Dict[str, TripleRawOutput], List[str]]:
                - Dictionary mapping chunk IDs to triplet outputs
                - List of entity+description pairs for embedding
        """
        # Extract passages from the provided chunks
        chunk_passages = {
            chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()
        }

        triplet_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create triplet extraction futures for each chunk
            triplet_futures = {
                executor.submit(self.extract_triplets, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            # Collect triplet extraction results with progress bar
            pbar = tqdm(
                as_completed(triplet_futures),
                total=len(triplet_futures),
                desc="Extracting triplets",
            )
            for future in pbar:
                result = future.result()
                triplet_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        # Now generate entity descriptions based on the extracted triplets
        entity_desc_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0

        with ThreadPoolExecutor() as executor:
            # Create entity description futures for each chunk
            desc_futures = {
                executor.submit(
                    self.generate_entity_descriptions,
                    triplet_result.chunk_id,
                    triplet_result.triples,
                    chunk_passages[triplet_result.chunk_id],
                ): triplet_result.chunk_id
                for triplet_result in triplet_results_list
            }

            # Collect entity description results with progress bar
            pbar = tqdm(
                as_completed(desc_futures),
                total=len(desc_futures),
                desc="Generating entity descriptions",
            )
            for future in pbar:
                result = future.result()
                entity_desc_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1
                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        # Create dictionaries for results
        triplet_results_dict = {res.chunk_id: res for res in triplet_results_list}
        entity_desc_results_dict = {
            res.chunk_id: res for res in entity_desc_results_list
        }

        # Create final combined triplet outputs
        final_triplet_results = {}
        # Collect all entity+description pairs for embedding
        all_entity_desc_pairs = []

        for chunk_id, triplet_output in triplet_results_dict.items():
            entity_desc_output = entity_desc_results_dict.get(chunk_id)

            if entity_desc_output:
                # Entity+description pairs for embedding
                entity_desc_pairs = []

                for (
                    entity,
                    description,
                ) in entity_desc_output.entity_descriptions.items():
                    # Combine entity and description for embedding
                    entity_with_desc = f"{entity}: {description}"
                    entity_desc_pairs.append(entity_with_desc)

                # Collect entity+description pairs for embedding
                all_entity_desc_pairs.extend(entity_desc_pairs)

                # Use original triplets only - no need to add entity-chunk triplets
                all_triplets = triplet_output.triples

                # Create updated triplet output
                final_triplet_results[chunk_id] = TripleRawOutput(
                    chunk_id=triplet_output.chunk_id,
                    response=triplet_output.response,
                    triples=all_triplets,
                    metadata=triplet_output.metadata,
                )
            else:
                # If no entity descriptions, use original triplets
                final_triplet_results[chunk_id] = triplet_output

        return final_triplet_results, all_entity_desc_pairs
