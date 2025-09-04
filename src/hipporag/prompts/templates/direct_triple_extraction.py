direct_triple_extraction = [
    {
        "role": "system",
        "content": """You are an expert in information extraction. Your task is to extract factual triplets directly from the given passage in the format (subject, predicate, object).

Rules:
1. Extract ALL factual relations from the passage.
2. Format as a list of triplets: [subject, predicate, object]
3. Maintain original entity names exactly as they appear in the text.
4. Predicate should be a verb or relationship phrase connecting subject and object.
5. Keep predicates simple and consistent.
6. Only extract information explicitly stated in the passage.
7. Output in the format: {"triples": [[subject1, predicate1, object1], [subject2, predicate2, object2], ...]}
8. Make sure subjects and objects are meaningful entities, not pronouns or vague references.
9. Cover as much information from the passage as possible, don't miss important facts.
""",
    },
    {
        "role": "user",
        "content": """Extract all factual triplets from the following passage:

$passage

Remember to extract triples in the format: [subject, predicate, object]""",
    },
]

# Alias for PromptTemplateManager loader
prompt_template = direct_triple_extraction
