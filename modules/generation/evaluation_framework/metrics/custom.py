CUSTOM_SYSTEM_PROMPT = """You are an expert evaluator for text generation quality. You will evaluate answers based on specific metrics and criteria provided."""

CUSTOM_INSTRUCTION_PROMPT = """You are evaluating the following based on a custom metric.

Metric Name: {metric_name}
Metric Description: {metric_description}
Evaluation Criteria:
{metric_criterion}

Question:
{question}

Answer:
{answer}

Ground Truth (if available):
{ground_truth}

Please evaluate according to the specified metric and criteria."""