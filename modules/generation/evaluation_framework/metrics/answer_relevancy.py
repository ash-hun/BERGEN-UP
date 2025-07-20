ANSWER_RELEVANCY_INSTRUCTION_PROMPT = """You are evaluating the relevancy of an answer to a given question.

Question:
{question}

Answer:
{answer}

Evaluation Criteria:
1. Does the answer directly address the question asked?
2. Is the answer complete and comprehensive?
3. Does the answer stay focused on the question without unnecessary information?

Please evaluate the relevancy of the answer on a scale from 1 to 5, where:
- 1: The answer is completely irrelevant to the question
- 2: The answer barely addresses the question
- 3: The answer partially addresses the question
- 4: The answer mostly addresses the question with minor gaps
- 5: The answer perfectly addresses the question"""