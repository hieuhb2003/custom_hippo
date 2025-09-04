entity_description = [
    {
        "role": "system",
        "content": """You are an expert in entity description generation. For the given entity name, create a concise, factual description based ONLY on information explicitly found in the provided passage. 

Rules:
1. The entity name in your response MUST match the original entity name exactly.
2. Only include information that is directly stated in or can be directly inferred from the passage.
3. Do not add any speculation, general knowledge, or information not contained in the passage.
4. Keep descriptions concise (1-3 sentences).
5. Format your response as a JSON object with the format: {"entity": "EXACT_ENTITY_NAME", "description": "ENTITY_DESCRIPTION"}
6. If the entity is not clearly described in the passage, provide a minimal description like "Entity mentioned in the document".
""",
    },
    {
        "role": "user",
        "content": """Given the entity: $entity

Passage:
$passage

Generate a concise description for this entity based only on information from the passage.""",
    },
]

# Alias for PromptTemplateManager loader
prompt_template = entity_description
