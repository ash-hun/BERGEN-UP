from typing import Optional, List, Dict, Any
from rich.console import Console
import pandas as pd

from modules.utils import rich_display_dataframe
from modules.pre_retrieval.evaluation_framework import (
    HyDEEvaluation,
    MultiQueryEvaluation,
    DecompositionEvaluation
)


class PreRetrievalEvaluation:
    '''
    🔄 Pre-retrieval Evaluation Module 🔄
    
    Evaluates pre-retrieval strategies using LLM-as-a-judge:
    - Multi-Query: Diversity, Coverage, Relevance
    - Query Decomposition: Completeness, Granularity, Independence, Answerability
    - HyDE: Relevance, Specificity, Factuality, Coherence
    '''
    def __init__(
        self, 
        pre_retrieval_strategy: List[dict],
        openai_api_key: Optional[str] = None
    ) -> None:
        self.pre_retrieval_strategy = pre_retrieval_strategy
        self.openai_api_key = openai_api_key
        self.console = Console()
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for Pre-retrieval evaluation")
        
        # Initialize evaluators
        self.evaluators = {
            'multi_query': MultiQueryEvaluation(openai_api_key),
            'decomposition': DecompositionEvaluation(openai_api_key),
            'hyde': HyDEEvaluation(openai_api_key)
        }
        
        # Extract strategy info
        self._parse_strategies()
    
    def __str__(self) -> str:
        return "🔄 Pre-retrieval Evaluation Module 🔄"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _parse_strategies(self) -> None:
        """Parse strategies from configuration."""
        self.strategies_to_eval = []
        
        for strategy in self.pre_retrieval_strategy:
            if 'Multi Query' in strategy:
                self.strategies_to_eval.append(('multi_query', strategy['Multi Query']['path']))
            elif 'Query Decomposition' in strategy:
                self.strategies_to_eval.append(('decomposition', strategy['Query Decomposition']['path']))
            elif 'HyDE' in strategy:
                self.strategies_to_eval.append(('hyde', strategy['HyDE']['path']))
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run pre-retrieval evaluation.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation results for each strategy
        """
        if verbose:
            self.console.log("🔄 Starting Pre-retrieval Evaluation", style="bold blue")
            self.console.log(f"Strategies to evaluate: {[s[0] for s in self.strategies_to_eval]}", style="blue")
        
        all_results = {}
        
        for strategy_type, data_path in self.strategies_to_eval:
            if verbose:
                self.console.log(f"\n📊 Evaluating {strategy_type} strategy", style="bold cyan")
            
            try:
                # Get the appropriate evaluator
                evaluator = self.evaluators[strategy_type]
                
                # Load sample data
                if verbose:
                    self.console.log(f"Loading data from: {data_path}", style="blue")
                data = evaluator.load_sample_data(data_path)
                
                # Run evaluation
                results = evaluator.evaluate(data, verbose=verbose)
                
                # Store results
                all_results[strategy_type] = results
                
                # Display results
                self._display_results(strategy_type, results)
                
                if verbose:
                    self.console.log(f"✅ {strategy_type} evaluation completed", style="bold green")
                
            except Exception as e:
                if verbose:
                    self.console.log(f"❌ {strategy_type} evaluation failed: {str(e)}", style="bold red")
                raise e
        
        return all_results
    
    def _display_results(self, strategy_type: str, results: Dict[str, Dict[str, float]]) -> None:
        """
        Display evaluation results in a formatted table.
        
        Args:
            strategy_type: Type of strategy evaluated
            results: Results dictionary from evaluation
        """
        # Calculate averages
        metric_averages = {}
        metric_values = {}
        
        # Collect all metric values
        for uuid, metrics in results.items():
            for metric_name, value in metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Calculate averages
        for metric_name, values in metric_values.items():
            if values:
                metric_averages[metric_name] = sum(values) / len(values)
        
        # Create display data
        display_data = []
        
        # Add header row with strategy type
        if strategy_type == 'multi_query':
            display_data.append({
                'Strategy': 'Multi-Query',
                'Diversity': f"{metric_averages.get('diversity', 0):.4f}",
                'Coverage': f"{metric_averages.get('coverage', 0):.4f}",
                'Relevance': f"{metric_averages.get('relevance', 0):.4f}"
            })
        elif strategy_type == 'decomposition':
            display_data.append({
                'Strategy': 'Query Decomposition',
                'Completeness': f"{metric_averages.get('completeness', 0):.4f}",
                'Granularity': f"{metric_averages.get('granularity', 0):.4f}",
                'Independence': f"{metric_averages.get('independence', 0):.4f}",
                'Answerability': f"{metric_averages.get('answerability', 0):.4f}"
            })
        elif strategy_type == 'hyde':
            display_data.append({
                'Strategy': 'HyDE',
                'Relevance': f"{metric_averages.get('relevance', 0):.4f}",
                'Specificity': f"{metric_averages.get('specificity', 0):.4f}",
                'Factuality': f"{metric_averages.get('factuality', 0):.4f}",
                'Coherence': f"{metric_averages.get('coherence', 0):.4f}"
            })
        
        # Convert to DataFrame and display
        df = pd.DataFrame(display_data)
        
        self.console.log(f"\n📊 {strategy_type.replace('_', ' ').title()} Results:", style="bold yellow")
        rich_display_dataframe(df, title=f"{strategy_type.replace('_', ' ').title()} Evaluation Metrics")
        
        # Display summary
        self.console.log("\n📈 Average Scores:", style="bold cyan")
        for metric, avg in metric_averages.items():
            self.console.log(f"{metric.capitalize()}: {avg:.4f}")