from modules.generation.evaluation_framework.utils.datamodel import GEvalResponse, MetricResult
from modules.generation.evaluation_framework.metrics.custom import CUSTOM_SYSTEM_PROMPT, CUSTOM_INSTRUCTION_PROMPT
from modules.generation.evaluation_framework.metrics.answer_relevancy import ANSWER_RELEVANCY_INSTRUCTION_PROMPT
from modules.generation.evaluation_framework.metrics.consistency import CONSISTENCY_INSTRUCTION_PROMPT
from modules.generation.evaluation_framework.metrics.fluency import FLUENCY_INSTRUCTION_PROMPT
from modules.generation.evaluation_framework.metrics.groundness import GROUNDNESS_INSTRUCTION_PROMPT
from modules.generation.evaluation_framework.metrics.relevancy import RELEVANCY_INSTRUCTION_PROMPT
from typing import Dict, Literal, List, Any, Optional
import instructor
import json
import uuid
import os
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from rich.console import Console

class BaseLLM:
    """ Base Class for LLM as a Judge """
    def __init__(self, conf: Dict, mode: Literal['custom', 'standard']) -> None:
        self.config = conf
        self.mode = mode
        self.console = Console()
    
    def _build_standard_prompt(self, metric: Literal['Answer Relevancy', 'Consistency', 'Fluency', 'Groundness', 'Relevancy'], 
                              question: str, answer: str, context: Optional[List[str]] = None, ground_truth: Optional[str] = None) -> str:
        """ Build standard prompt """
        if metric == 'Answer Relevancy':
            return ANSWER_RELEVANCY_INSTRUCTION_PROMPT.format(
                question=question,
                answer=answer
            )
        elif metric == 'Consistency':
            return CONSISTENCY_INSTRUCTION_PROMPT.format(
                answer=answer
            )
        elif metric == 'Fluency':
            return FLUENCY_INSTRUCTION_PROMPT.format(
                answer=answer
            )
        elif metric == 'Groundness':
            context_text = "\n".join(context) if context else ""
            return GROUNDNESS_INSTRUCTION_PROMPT.format(
                context=context_text,
                answer=answer
            )
        elif metric == 'Relevancy':
            context_text = "\n".join(context) if context else ""
            return RELEVANCY_INSTRUCTION_PROMPT.format(
                question=question,
                context=context_text
            )
        else:
            raise ValueError(f"🔥 Unsupported metric: {metric}, please check the metric name!")
    
    def _build_custom_prompt(self, question: str, answer: str, ground_truth: str = "") -> str:
        """ Build custom prompt """
        return CUSTOM_INSTRUCTION_PROMPT.format(
            metric_name=self.config["metric_name"],
            metric_description=self.config["metric_description"],
            metric_criterion=self.config["metric_criterion"],
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )

    def _get_instructor_client(self, api_key: Optional[str] = None) -> object:
        """ Get instructor client for structured output """
        model_name = self.config.get("metric_llm", {}).get("model_name", "gpt-4")
        
        # Use provided API key or get from environment
        if api_key:
            openai_api_key = api_key
        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if not openai_api_key:
            raise ValueError("🔥 OpenAI API key is required for G-Eval")
            
        # For BERGEN-UP, we'll use OpenAI models
        client = OpenAI(api_key=openai_api_key)
        return instructor.from_openai(client)
    
class Evaluator(BaseLLM):
    """ LLM as a Judge """
    def __init__(self, conf: dict, mode: Literal['custom', 'standard'], api_key: Optional[str] = None):
        super().__init__(conf, mode)
        self.config = conf
        self.mode = mode
        self.api_key = api_key
    
    def _load_dataset(self) -> dict:
        """ Load dataset from JSON file """
        dataset_path = self.config.get("dataset_path")
        if not dataset_path:
            raise ValueError("Dataset path not provided")
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'sample_results' in data:
            return data['sample_results']
        else:
            return data  # Return as is for BERGEN-UP format

    def evaluate(self, client: object, prompt: str) -> MetricResult:
        """ Evaluate the prompt """
        model_name = self.config.get("metric_llm", {}).get("model_name", "gpt-4")
        temperature = self.config.get("metric_llm", {}).get("temperature", 0.0)
        max_tokens = self.config.get("metric_llm", {}).get("max_tokens", 1024)
        
        return client.chat.completions.create(
            model=model_name,
            response_model=MetricResult,
            messages=[
                {"role": "system", "content": CUSTOM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def run(self, data: Optional[Dict[str, Any]] = None, verbose: bool = True) -> GEvalResponse:
        # Get instructor client
        client = self._get_instructor_client(self.api_key)
        
        # Load dataset if not provided
        if data is None:
            dataset = self._load_dataset()
        else:
            dataset = data

        # Process all samples and collect results
        all_results = []
        all_scores = 0
        
        # Handle different data formats
        if isinstance(dataset, list):
            # List format (from G-Eval sample files)
            samples = dataset
        elif isinstance(dataset, dict) and 'question' in dataset:
            # BERGEN-UP format
            samples = []
            for uuid_key in dataset['question'].keys():
                sample = {
                    "question_id": uuid_key,
                    "question": dataset['question'][uuid_key],
                    "predicted": dataset.get('answer', {}).get(uuid_key, ""),
                    "context": dataset.get('context', {}).get(uuid_key, []),
                    "ground_truth": dataset.get('ground_truth', {}).get(uuid_key, "")
                }
                samples.append(sample)
        else:
            raise ValueError("Unsupported dataset format")
        
        # Use tqdm if verbose is True
        iterator = tqdm(samples, desc="Evaluating...") if verbose else samples
        
        for sample in iterator:
            # Build evaluation prompt for each sample
            if self.mode == 'custom':
                prompt = self._build_custom_prompt(
                    question=sample.get("question", ""),
                    answer=sample.get("predicted", sample.get("answer", "")),
                    ground_truth=sample.get("ground_truth", "")
                )
            elif self.mode == 'standard':
                prompt = self._build_standard_prompt(
                    metric=self.config["metric_name"],
                    question=sample.get("question", ""),
                    answer=sample.get("predicted", sample.get("answer", "")),
                    context=sample.get("context", []),
                    ground_truth=sample.get("ground_truth", "")
                )
            
            # Perform structured evaluation
            result = self.evaluate(client, prompt)
            normalized_score = result.score / 5.0  # Normalize 1-5 to 0-1
            all_scores += normalized_score
            
            all_results.append({
                "question_id": sample.get("question_id", ""),
                "question": sample.get("question", ""),
                "predicted": sample.get("predicted", sample.get("answer", "")),
                "ground_truth": sample.get("ground_truth", ""),
                "metric_score": normalized_score,
                "metric_explanation": result.reasoning
            })
        
        # Calculate average score
        avg_score = all_scores / len(all_results) if all_results else 0.0
        
        # Return structured response with all results
        return GEvalResponse(
            metric_name=self.config["metric_name"],
            metric_score=avg_score,
            metric_explanation=f"Evaluated {len(all_results)} samples with average score: {avg_score:.2f}",
            metric_evaluation_id=str(uuid.uuid4()),
            metric_timestamp=datetime.now(),
            detailed_results=all_results
        )