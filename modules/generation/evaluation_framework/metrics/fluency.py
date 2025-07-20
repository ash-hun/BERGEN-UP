FLUENCY_INSTRUCTION_PROMPT = """You are evaluating the fluency and readability of an answer.

Answer:
{answer}

Evaluation Criteria:
1. Is the answer grammatically correct?
2. Does the answer flow naturally?
3. Is the answer easy to read and understand?

Please evaluate the fluency of the answer on a scale from 1 to 5, where:
- 1: The answer is very difficult to read with many grammatical errors
- 2: The answer has significant fluency issues
- 3: The answer is somewhat readable with noticeable issues
- 4: The answer is mostly fluent with minor issues
- 5: The answer is perfectly fluent and easy to read"""