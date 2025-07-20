from typing import Optional, List, Dict, Any
from rich.console import Console
import pandas as pd
import json
from pathlib import Path

from modules.utils import rich_display_dataframe
from modules.generation.evaluation_framework.g_eval import GEval

class GenerationEvaluation:
    '''
    🤖 Generation Evaluation Module 🤖
    
    Evaluates generation quality using G-Eval with metrics:
    - Groundedness: How well the answer is grounded in the provided context
    - Answer Relevancy: How relevant the answer is to the question
    - Custom metrics: User-defined evaluation criteria
    
    Input format:
    {
        'question': Dict[uuid, str],
        'context': Dict[uuid, List[str]], 
        'answer': Dict[uuid, str]
    }
    
    Output format:
    {
        'uuid-1': {'groundedness': 0.9, 'answer_relevancy': 0.95},
        ...
    }
    '''
    def __init__(
        self, 
        generation_strategy: List[dict],
        openai_api_key: Optional[str] = None
    ) -> None:
        self.generation_strategy = generation_strategy
        self.openai_api_key = openai_api_key
        self.console = Console()
        
        # Extract configuration from strategies
        self.sample_data_path = None
        self.evaluation_metrics = ['groundedness', 'answer_relevancy']  # Default metrics
        self.g_eval_config = None
        
        for strategy in generation_strategy:
            if 'sample_data_path' in strategy:
                self.sample_data_path = strategy['sample_data_path']
            if 'evaluation_metrics' in strategy:
                self.evaluation_metrics = strategy['evaluation_metrics']
            if 'g_eval_config' in strategy:
                self.g_eval_config = strategy['g_eval_config']
        
        # Initialize G-Eval evaluator
        if openai_api_key:
            self.evaluator = GEval(openai_api_key)
        else:
            raise ValueError("OpenAI API key is required for Generation evaluation")
    
    def __str__(self) -> str:
        return "🤖 Generation Evaluation Module 🤖"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, data: Dict[str, Any] = None, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Run generation evaluation.
        
        Args:
            data: Input data dictionary with format:
                {
                    'question': Dict[uuid, str],
                    'context': Dict[uuid, List[str]], 
                    'answer': Dict[uuid, str]
                }
                If None, loads sample data from data/generation/sample_data.json.
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        # Load sample data if none provided
        if data is None:
            if verbose:
                self.console.log("🤖 No data provided, loading sample data from JSON file", style="bold blue")
            data = self._load_sample_data()
        
        if verbose:
            self.console.log("🤖 Starting Generation Evaluation", style="bold blue")
            self.console.log(f"Number of questions: {len(data['question'])}", style="blue")
            
        try:
            # Check if G-Eval config is provided
            if self.g_eval_config:
                mode = self.g_eval_config.get('mode', 'standard')
                
                if mode == 'standard':
                    # Use standard metric evaluation
                    metric_name = self.g_eval_config.get('metric_name', 'Answer Relevancy')
                    model_name = self.g_eval_config.get('metric_llm', {}).get('model_name', 'gpt-4')
                    temperature = self.g_eval_config.get('metric_llm', {}).get('temperature', 0.0)
                    
                    if verbose:
                        self.console.log(f"Using G-Eval standard mode with metric: {metric_name}", style="blue")
                    
                    response = self.evaluator.evaluate_with_metric(
                        metric_name=metric_name,
                        data=data,
                        model_name=model_name,
                        temperature=temperature
                    )
                    
                    # Convert response to results format
                    results = {}
                    for i, result in enumerate(response.detailed_results):
                        uuid = result['question_id']
                        results[uuid] = {
                            metric_name.lower().replace(' ', '_'): result['metric_score']
                        }
                    
                    # Display G-Eval results
                    self._display_g_eval_results(response)
                    
                elif mode == 'custom':
                    # Use custom metric evaluation
                    metric_name = self.g_eval_config.get('metric_name', 'Custom Metric')
                    metric_description = self.g_eval_config.get('metric_description', '')
                    metric_criterion = self.g_eval_config.get('metric_criterion', '')
                    model_name = self.g_eval_config.get('metric_llm', {}).get('model_name', 'gpt-4')
                    temperature = self.g_eval_config.get('metric_llm', {}).get('temperature', 0.0)
                    
                    if verbose:
                        self.console.log(f"Using G-Eval custom mode with metric: {metric_name}", style="blue")
                    
                    response = self.evaluator.evaluate_custom(
                        metric_name=metric_name,
                        metric_description=metric_description,
                        metric_criterion=metric_criterion,
                        data=data,
                        model_name=model_name,
                        temperature=temperature
                    )
                    
                    # Convert response to results format
                    results = {}
                    for i, result in enumerate(response.detailed_results):
                        uuid = result['question_id']
                        results[uuid] = {
                            'custom_metric': result['metric_score']
                        }
                    
                    # Display G-Eval results
                    self._display_g_eval_results(response)
                    
                else:
                    raise ValueError(f"Invalid G-Eval mode: {mode}")
                    
            else:
                # Use default evaluation (backward compatibility)
                if verbose:
                    self.console.log("Using default evaluation metrics: Groundedness, Answer Relevancy", style="blue")
                    
                results = self.evaluator.evaluate_batch(data, verbose=verbose)
                
                # Always display results regardless of verbose setting
                self._display_results(results)
            
            if verbose:
                self.console.log("✅ Evaluation completed successfully", style="bold green")
            
            return results
            
        except Exception as e:
            if verbose:
                self.console.log(f"❌ Evaluation failed: {str(e)}", style="bold red")
            raise e
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """
        Load sample data from the configured JSON file.
        
        Returns:
            Dictionary with sample data
        """
        if not self.sample_data_path:
            raise ValueError("No sample data path configured in generation strategies")
        
        # Handle Hydra variable interpolation
        sample_path = self.sample_data_path
        if '${hydra:runtime.cwd}' in sample_path:
            import os
            sample_path = sample_path.replace('${hydra:runtime.cwd}', os.getcwd())
        
        sample_data_file = Path(sample_path)
        if not sample_data_file.exists():
            raise FileNotFoundError(f"Sample data file not found: {sample_data_file}")
        
        with open(sample_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.console.log(f"📂 Loaded sample data from: {sample_data_file}", style="green")
        return data
    
    def _display_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Display evaluation results in a formatted table.
        
        Args:
            results: Results dictionary from evaluation
        """
        # Calculate averages
        groundedness_scores = []
        relevancy_scores = []
        
        for uuid, metrics in results.items():
            groundedness_scores.append(metrics['groundedness'])
            relevancy_scores.append(metrics['answer_relevancy'])
        
        # Create DataFrame with only average
        avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        
        display_data = [{
            'Metric': 'Average',
            'Groundedness': f"{avg_groundedness:.4f}",
            'Answer Relevancy': f"{avg_relevancy:.4f}"
        }]
        
        # Convert to DataFrame and display
        df = pd.DataFrame(display_data)
        
        self.console.log("\n📊 Generation Evaluation Results:", style="bold yellow")
        rich_display_dataframe(df, title="Generation Quality Metrics")
        
        # Display summary
        self.console.log("\n📈 Summary:", style="bold cyan")
        self.console.log(f"Average Groundedness: {avg_groundedness:.4f}")
        self.console.log(f"Average Answer Relevancy: {avg_relevancy:.4f}")
    
    def _display_g_eval_results(self, response) -> None:
        """
        Display G-Eval evaluation results in a formatted table.
        
        Args:
            response: GEvalResponse object from evaluation
        """
        # Create DataFrame for display
        display_data = [{
            'Metric': response.metric_name,
            'Average Score': f"{response.metric_score:.4f}",
            'Total Samples': len(response.detailed_results) if response.detailed_results else 0
        }]
        
        # Convert to DataFrame and display
        df = pd.DataFrame(display_data)
        
        self.console.log("\n📊 G-Eval Results:", style="bold yellow")
        rich_display_dataframe(df, title="G-Eval Metrics")
        
        # Display summary
        self.console.log("\n📈 Summary:", style="bold cyan")
        self.console.log(f"Evaluation ID: {response.metric_evaluation_id}")
        self.console.log(f"Timestamp: {response.metric_timestamp}")
        self.console.log(f"Explanation: {response.metric_explanation}")