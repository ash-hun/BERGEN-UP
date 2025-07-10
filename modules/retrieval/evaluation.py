from typing import Optional, List, Dict, Any
from rich.console import Console
import pandas as pd
import json
from pathlib import Path

from modules.utils import rich_display_dataframe
from modules.retrieval.evaluation_framework.general_evaluation import GeneralRetrievalEvaluation

class RetrievalEvaluation:
    '''
    🔍 Retrieval Evaluation Module 🔍
    
    Evaluates retrieval performance using various metrics:
    - Precision@k, Recall@k, F1@k
    - MRR (Mean Reciprocal Rank)
    - NDCG@k (Normalized Discounted Cumulative Gain)
    - Hit Rate@k
    
    Input format:
    {
        'question': Dict[uuid, str],
        'context_gt_map': Dict[uuid, int], 
        'context_gt': Dict[int, str],
        'retrieved': Dict[uuid, List[str]]
    }
    
    Output format:
    {
        'top_k': Dict[metric, score],
        ...
    }
    '''
    def __init__(
            self, 
            retrieval_strategy: List[dict],
            openai_api_key: Optional[str] = None,
            top_k_values: List[int] = None
        ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.openai_api_key = openai_api_key
        self.top_k_values = top_k_values or [1, 3, 5, 10, 20]
        self.console = Console()
        self.evaluator = GeneralRetrievalEvaluation(top_k_values=self.top_k_values)
        
        # Extract sample data path from strategies (following chunking module pattern)
        self.sample_data_path = retrieval_strategy[0]['sample_data_path'] if retrieval_strategy and 'sample_data_path' in retrieval_strategy[0] else None
    
    def __str__(self) -> str:
        return "🔍 Retrieval Evaluation Module 🔍"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, data: Dict[str, Any] = None, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Run retrieval evaluation.
        
        Args:
            data: Input data dictionary with format:
                {
                    'question': Dict[uuid, str],
                    'context_gt_map': Dict[uuid, int], 
                    'context_gt': Dict[int, str],
                    'retrieved': Dict[uuid, List[str]]
                }
                If None, uses sample data for demonstration.
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with evaluation results:
            {
                'top_1': {'precision': 0.8, 'recall': 0.2, ...},
                'top_3': {'precision': 0.7, 'recall': 0.4, ...},
                ...
                'mrr': {'mrr': 0.65}
            }
        """
        # Load sample data if none provided
        if data is None:
            if verbose:
                self.console.log("🔍 No data provided, loading sample data from JSON file", style="bold blue")
            data = self._load_sample_data()
        
        # Convert retrieved keys to actual text content for evaluation
        data = self._convert_retrieved_keys_to_text(data)
        
        if verbose:
            self.console.log("🔍 Starting Retrieval Evaluation", style="bold blue")
            self.console.log(f"Number of questions: {len(data['question'])}", style="blue")
            self.console.log(f"Top-k values: {self.top_k_values}", style="blue")
        
        try:
            # Run evaluation
            results = self.evaluator.evaluate(data)
            
            if verbose:
                self.console.log("✅ Evaluation completed successfully", style="bold green")
            
            # Always display results regardless of verbose setting
            self._display_results(results)
            
            return results
            
        except Exception as e:
            if verbose:
                self.console.log(f"❌ Evaluation failed: {str(e)}", style="bold red")
            raise e
    
    def _display_results(self, summary:bool=True, results: Dict[str, Dict[str, float]]=None) -> None:
        """
        Display evaluation results in a formatted table.
        
        Args:
            results: Results dictionary from evaluation
        """
        # Create DataFrame for display
        display_data = []
        
        # Get MRR value if exists
        mrr_value = results.get('mrr', {}).get('mrr', None)
        
        # Add top-k results with MRR
        for top_k_key in sorted(results.keys(), key=lambda x: int(x.replace('top_', '')) if x.startswith('top_') else 0):
            if top_k_key.startswith('top_'):
                k_value = int(top_k_key.replace('top_', ''))
                metrics = results[top_k_key]
                
                row = {'Top-K': k_value}
                # Add all metrics from results
                for metric, score in metrics.items():
                    row[metric.capitalize()] = f"{score:.4f}"
                
                # Add MRR value (same for all rows)
                if mrr_value is not None:
                    row['MRR'] = f"{mrr_value:.4f}"
                else:
                    row['MRR'] = '-'
                
                display_data.append(row)
        
        # Convert to DataFrame and display
        df = pd.DataFrame(display_data)
        
        self.console.log("\n📊 Retrieval Evaluation Results:", style="bold yellow")
        rich_display_dataframe(df, title="Retrieval Performance Metrics")
        
        # Display summary
        if summary:
            summary = self.evaluator.get_summary_report(results)
            self.console.log("\n📈 Summary:", style="bold cyan")
            self.console.log(f"Best Precision: {summary['best_precision']['value']:.4f} @k={summary['best_precision']['at_k']}")
            self.console.log(f"Best Recall: {summary['best_recall']['value']:.4f} @k={summary['best_recall']['at_k']}")
            self.console.log(f"Best F1: {summary['best_f1']['value']:.4f} @k={summary['best_f1']['at_k']}")
            self.console.log(f"Best NDCG: {summary['best_ndcg']['value']:.4f} @k={summary['best_ndcg']['at_k']}")
            self.console.log(f"Best Hit Rate: {summary['best_hit_rate']['value']:.4f} @k={summary['best_hit_rate']['at_k']}")
            self.console.log(f"MRR: {summary['mrr']:.4f}")
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """
        Load sample data from the configured JSON file.
        
        Returns:
            Dictionary with sample data
        """
        if not self.sample_data_path:
            raise ValueError("No sample data path configured in retrieval strategies")
        
        sample_data_file = Path(self.sample_data_path)
        if not sample_data_file.exists():
            raise FileNotFoundError(f"Sample data file not found: {sample_data_file}")
        
        with open(sample_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string keys back to integers for context_gt
        if 'context_gt' in data:
            data['context_gt'] = {int(k): v for k, v in data['context_gt'].items()}
        
        self.console.log(f"📂 Loaded sample data from: {sample_data_file}", style="green")
        return data
    
    def _convert_retrieved_keys_to_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert retrieved key values to actual text content for evaluation.
        
        Args:
            data: Data dictionary with retrieved keys
            
        Returns:
            Data dictionary with retrieved text content
        """
        converted_data = data.copy()
        
        if 'retrieved' in data and 'context_gt' in data:
            context_gt = data['context_gt']
            converted_retrieved = {}
            
            for uuid, key_list in data['retrieved'].items():
                # Convert each key to corresponding text content
                text_list = []
                for key in key_list:
                    if key in context_gt:
                        text_list.append(context_gt[key])
                    else:
                        # If key doesn't exist, use placeholder
                        text_list.append(f"Content not found for key: {key}")
                
                converted_retrieved[uuid] = text_list
            
            converted_data['retrieved'] = converted_retrieved
        
        return converted_data