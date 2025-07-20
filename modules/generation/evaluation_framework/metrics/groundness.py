GROUNDNESS_INSTRUCTION_PROMPT = """You are evaluating the groundedness of an answer based on the given context.

Context:
{context}

Answer:
{answer}

Evaluation Criteria:
1. Is the answer fully supported by the context? 
2. Does the answer contain any information not present in the context?
3. Are all claims in the answer verifiable from the context?

Please evaluate the groundedness of the answer on a scale from 1 to 5, where:
- 1: The answer is completely ungrounded (contains information not in context)
- 2: The answer is mostly ungrounded with some correct information
- 3: The answer is partially grounded (some information from context, some not)
- 4: The answer is mostly grounded with minor unsupported claims
- 5: The answer is fully grounded (all information comes from context)"""