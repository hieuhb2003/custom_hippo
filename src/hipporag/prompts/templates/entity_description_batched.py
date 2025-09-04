prompt_template = [
    {
        "role": "system",
        "content": """You are an expert in entity description generation. Generate concise factual descriptions ONLY from the given passage.

Rules:
1. Output a JSON object mapping each entity to its description.
2. Keep each description 1-2 sentences, factual and grounded in the passage.
3. If the passage does not contain enough info, use: \"Entity mentioned in the document\".
4. Maintain entity names EXACTLY as provided.
""",
    },
    {
        "role": "user",
        "content": """Passage:
$passage

Entities:
$entities_json

Optional context triples (extracted):
$triples_json

Task: Return a JSON object of the form {\"ENTITY\": \"DESCRIPTION\", ...} covering ALL entities above. Use the passage and triples to avoid missing entities.""",
    },
]
