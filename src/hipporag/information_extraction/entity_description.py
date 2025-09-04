import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json

logger = get_logger(__name__)


@dataclass
class EntityDescription:
    entity: str
    description: str


@dataclass
class EntityDescriptionOutput:
    chunk_id: str
    entity_descriptions: Dict[str, str]
    metadata: Dict
    entity_desc_pairs: List[str] = field(default_factory=list)


class EntityDescriptionGenerator:
    """
    Class responsible for generating descriptions for entities found in triplets.
    """

    def __init__(self, llm_model):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def generate_description(self, entity: str, passage: str) -> EntityDescription:
        """
        Generate a description for a single entity based on the passage.

        Args:
            entity (str): The entity name to generate description for
            passage (str): The passage containing information about the entity

        Returns:
            EntityDescription: Entity name and its description
        """
        # PREPROCESSING
        desc_input_message = self.prompt_template_manager.render(
            name="entity_description", entity=entity, passage=passage
        )

        raw_response = ""
        metadata = {}

        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=desc_input_message,
            )
            metadata["cache_hit"] = cache_hit

            if metadata["finish_reason"] == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            # Extract description from response
            description = self._extract_description_from_response(real_response, entity)

            return EntityDescription(entity=entity, description=description)

        except Exception as e:
            # For any unexpected exceptions, log them and return with minimal description
            logger.warning(f"Error generating description for entity '{entity}': {e}")
            return EntityDescription(
                entity=entity, description="Entity found in document"
            )

    def _extract_description_from_response(self, response: str, entity: str) -> str:
        """
        Extract the description from the LLM response.

        Args:
            response (str): The raw LLM response
            entity (str): The entity name (for fallback)

        Returns:
            str: The extracted description or a fallback
        """
        try:
            # Try to parse JSON response
            if "{" in response and "}" in response:
                try:
                    # Find JSON object in response
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                    data = json.loads(json_str)

                    # Return description if it exists
                    if "description" in data:
                        return data["description"]
                except Exception:
                    pass

            # Simple extraction if JSON parsing fails
            if "description:" in response.lower():
                parts = response.lower().split("description:")
                if len(parts) > 1:
                    return parts[1].strip()

            # If all parsing fails, return the raw response
            return response.strip()

        except Exception as e:
            logger.warning(f"Error extracting description: {e}")
            return "Entity found in document"

    def batch_generate_descriptions(
        self, entities: Set[str], passage: str
    ) -> Tuple[Dict[str, str], Dict]:
        """
        Generate descriptions for multiple entities from the same passage in parallel.

        Args:
            entities (Set[str]): Set of unique entities to generate descriptions for
            passage (str): The passage containing information about the entities

        Returns:
            Dict[str, str]: Dictionary mapping entity names to their descriptions
        """
        descriptions = {}

        with ThreadPoolExecutor() as executor:
            # Create futures for each entity
            desc_futures = {
                executor.submit(self.generate_description, entity, passage): entity
                for entity in entities
            }

            # Collect results with progress bar
            pbar = tqdm(
                as_completed(desc_futures),
                total=len(desc_futures),
                desc="Generating entity descriptions",
            )
            for future in pbar:
                result = future.result()
                entity = result.entity
                descriptions[entity] = result.description

                # Update metrics for display
                pbar.set_postfix(
                    {
                        "total_entities": len(descriptions),
                        "remaining": len(desc_futures) - len(descriptions),
                    }
                )

        # For now, aggregate metadata is empty; individual call metadata is not combined here
        return descriptions, {}

    def extract_entities_from_triplets(self, triplets: List[List[str]]) -> Set[str]:
        """
        Extract unique entities from a list of triplets.

        Args:
            triplets (List[List[str]]): List of triplets [subject, predicate, object]

        Returns:
            Set[str]: Set of unique entities from subjects and objects
        """
        entities = set()
        for triplet in triplets:
            if len(triplet) == 3:
                # Add subject and object as entities
                entities.add(triplet[0])
                entities.add(triplet[2])
        return entities

    def generate_entity_description_triplets(
        self, triplets: List[List[str]], passage: str, chunk_id: str
    ) -> Tuple[List[List[str]], Dict[str, str]]:
        """
        Generate entity-description-document triplets from existing triplets.

        Args:
            triplets (List[List[str]]): Original triplets from the document
            passage (str): The document passage
            chunk_id (str): Identifier for the chunk/document

        Returns:
            Tuple[List[List[str]], Dict[str, str]]:
                - New triplets in the format [entity, "describes", description]
                - Dictionary mapping entities to their descriptions
        """
        # Extract unique entities from triplets
        entities = self.extract_entities_from_triplets(triplets)

        # Generate descriptions for all entities
        entity_descriptions = self.batch_generate_descriptions(entities, passage)

        # Create new triplets: entity, "describes", description
        description_triplets = []
        for entity, description in entity_descriptions.items():
            description_triplets.append([entity, "has_description", description])
            # Also add a triplet linking entity to document
            description_triplets.append([entity, "appears_in", chunk_id])

        return description_triplets, entity_descriptions
