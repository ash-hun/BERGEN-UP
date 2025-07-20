CONSISTENCY_INSTRUCTION_PROMPT = """You are evaluating the internal consistency of an answer.

Answer:
{answer}

Evaluation Criteria:
1. Does the answer contradict itself?
2. Are all statements in the answer logically consistent?
3. Does the answer maintain a coherent narrative throughout?

Please evaluate the consistency of the answer on a scale from 1 to 5, where:
- 1: The answer contains multiple contradictions
- 2: The answer has significant inconsistencies
- 3: The answer has some minor inconsistencies
- 4: The answer is mostly consistent with rare lapses
- 5: The answer is perfectly consistent throughout"""