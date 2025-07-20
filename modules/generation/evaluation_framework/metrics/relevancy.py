RELEVANCY_INSTRUCTION_PROMPT = """You are evaluating the overall relevancy of retrieved context to a given question.

Question:
{question}

Context:
{context}

Evaluation Criteria:
1. Does the context contain information relevant to answering the question?
2. How much of the context is actually useful for the question?
3. Is there sufficient information in the context to answer the question?

Please evaluate the relevancy of the context on a scale from 1 to 5, where:
- 1: The context is completely irrelevant to the question
- 2: The context has minimal relevance to the question
- 3: The context is somewhat relevant but missing key information
- 4: The context is mostly relevant with minor gaps
- 5: The context is perfectly relevant and complete"""