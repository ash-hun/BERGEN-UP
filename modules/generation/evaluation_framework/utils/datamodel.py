from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional

## Interface Level ##

class CustomMetricRequest(BaseModel):
    """ Custom Metric Evaluation for G-Eval """
    metric_name: str = "Awesome Metric"
    metric_description: str = "Evaluating how awesome [1-3] the answer is to the given question"
    metric_criterion: str = """
        - 1: Poor. The answer is not awesome.
        - 2: Fair. The answer is somewhat awesome.
        - 3: Good. The answer is very awesome.
    """
    metric_llm: dict = {
        "model_name": "gpt-4",
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    dataset_path: str = "./dataset/sample.json"

class StandardMetricRequest(BaseModel):
    """ Standard Metric Evaluation for G-Eval """
    metric_name: str
    metric_llm: dict
    dataset_path: str

class GEvalResponse(BaseModel):
    """ G-Eval Response Model """
    metric_name: str
    metric_score: float
    metric_explanation: str
    metric_evaluation_id: str
    metric_timestamp: datetime
    detailed_results: Optional[List[Dict[str, Any]]] = None

## Data Schema Level ##

class MetricResult(BaseModel):
    score: int = Field(ge=1, le=5, description="Evaluation score between 1 and 5")
    reasoning: str = Field(description="Brief explanation of the evaluation")