from typing import Dict, List, Any, Literal, Optional
import asyncio
import concurrent.futures
from rich.console import Console
from modules.generation.evaluation_framework.evaluate.llm_as_a_judge import Evaluator
from modules.generation.evaluation_framework.utils.datamodel import GEvalResponse

class GEval:
    """
    G-Eval implementation for generation evaluation.
    
    Supports both standard and custom metrics:
    - Standard metrics: Groundedness, Answer Relevancy, Consistency, Fluency, Relevancy
    - Custom metrics: User-defined evaluation criteria
    """
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.console = Console()
    
    def _run_evaluation_sync(self, evaluator, data, verbose=False):
        """Helper method to run evaluation in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(evaluator.run(data, verbose=verbose))
            return response
        finally:
            # Clean up any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    
    def evaluate_groundedness(self, context: List[str], answer: str) -> float:
        """
        Evaluate how well the answer is grounded in the provided context.
        
        Args:
            context: List of context sentences
            answer: Generated answer
            
        Returns:
            Groundedness score (0-1)
        """
        config = {
            "metric_name": "Groundness",
            "metric_llm": {
                "model_name": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1024
            }
        }
        
        evaluator = Evaluator(config, mode='standard', api_key=self.openai_api_key)
        
        # Create a single sample for evaluation
        data = {
            "question": {"sample": ""},
            "context": {"sample": context},
            "answer": {"sample": answer}
        }
        
        # Run evaluation synchronously
        try:
            # Check if we're already in an async context (like FastAPI)
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_evaluation_sync, evaluator, data, False)
                    response = future.result()
                    return response.metric_score
            except RuntimeError:
                # No current loop, safe to create a new one (CLI context)
                response = self._run_evaluation_sync(evaluator, data, False)
                return response.metric_score
        except Exception as e:
            self.console.print(f"[red]Error in evaluation: {e}[/red]")
            return 0.0
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question.
        
        Args:
            question: The question asked
            answer: Generated answer
            
        Returns:
            Relevancy score (0-1)
        """
        config = {
            "metric_name": "Answer Relevancy",
            "metric_llm": {
                "model_name": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1024
            }
        }
        
        evaluator = Evaluator(config, mode='standard', api_key=self.openai_api_key)
        
        # Create a single sample for evaluation
        data = {
            "question": {"sample": question},
            "answer": {"sample": answer}
        }
        
        # Run evaluation synchronously
        try:
            # Check if we're already in an async context (like FastAPI)
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_evaluation_sync, evaluator, data, False)
                    response = future.result()
                    return response.metric_score
            except RuntimeError:
                # No current loop, safe to create a new one (CLI context)
                response = self._run_evaluation_sync(evaluator, data, False)
                return response.metric_score
        except Exception as e:
            self.console.print(f"[red]Error in evaluation: {e}[/red]")
            return 0.0
    
    def evaluate_with_metric(
        self, 
        metric_name: Literal['Answer Relevancy', 'Consistency', 'Fluency', 'Groundness', 'Relevancy'],
        data: Dict[str, Any],
        model_name: str = "gpt-4",
        temperature: float = 0.0
    ) -> GEvalResponse:
        """
        Evaluate using a specific standard metric.
        
        Args:
            metric_name: Name of the metric to use
            data: Evaluation data
            model_name: LLM model to use for evaluation
            temperature: Temperature for LLM generation
            
        Returns:
            GEvalResponse with evaluation results
        """
        config = {
            "metric_name": metric_name,
            "metric_llm": {
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": 1024
            }
        }
        
        evaluator = Evaluator(config, mode='standard', api_key=self.openai_api_key)
        
        # Run evaluation
        try:
            # Check if we're already in an async context (like FastAPI)
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_evaluation_sync, evaluator, data, True)
                    response = future.result()
                    return response
            except RuntimeError:
                # No current loop, safe to create a new one (CLI context)
                response = self._run_evaluation_sync(evaluator, data, True)
                return response
        except Exception as e:
            self.console.print(f"[red]Error in evaluation: {e}[/red]")
            return None
    
    def evaluate_custom(
        self,
        metric_name: str,
        metric_description: str,
        metric_criterion: str,
        data: Dict[str, Any],
        model_name: str = "gpt-4",
        temperature: float = 0.0
    ) -> GEvalResponse:
        """
        Evaluate using custom metric definition.
        
        Args:
            metric_name: Custom metric name
            metric_description: Description of what the metric measures
            metric_criterion: Evaluation criteria (1-5 scale)
            data: Evaluation data
            model_name: LLM model to use for evaluation
            temperature: Temperature for LLM generation
            
        Returns:
            GEvalResponse with evaluation results
        """
        config = {
            "metric_name": metric_name,
            "metric_description": metric_description,
            "metric_criterion": metric_criterion,
            "metric_llm": {
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": 1024
            }
        }
        
        evaluator = Evaluator(config, mode='custom', api_key=self.openai_api_key)
        
        # Run evaluation
        try:
            # Check if we're already in an async context (like FastAPI)
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_evaluation_sync, evaluator, data, True)
                    response = future.result()
                    return response
            except RuntimeError:
                # No current loop, safe to create a new one (CLI context)
                response = self._run_evaluation_sync(evaluator, data, True)
                return response
        except Exception as e:
            self.console.print(f"[red]Error in evaluation: {e}[/red]")
            return None
    
    def evaluate_batch(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of generation results with standard metrics.
        This method maintains backward compatibility.
        
        Args:
            data: Dictionary containing questions, contexts, and answers
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation scores for each UUID
        """
        results = {}
        uuids = list(data['question'].keys())
        
        # Evaluate groundedness
        groundness_response = self.evaluate_with_metric(
            'Groundness',
            data,
            verbose=verbose
        )
        
        # Evaluate answer relevancy
        relevancy_response = self.evaluate_with_metric(
            'Answer Relevancy',
            data,
            verbose=verbose
        )
        
        # Combine results
        for i, uuid in enumerate(uuids):
            results[uuid] = {
                'groundedness': groundness_response.detailed_results[i]['metric_score'] if i < len(groundness_response.detailed_results) else 0.0,
                'answer_relevancy': relevancy_response.detailed_results[i]['metric_score'] if i < len(relevancy_response.detailed_results) else 0.0
            }
        
        return results