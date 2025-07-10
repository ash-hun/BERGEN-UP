from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from rich.console import Console
import json
from pathlib import Path


class BasePreRetrievalEvaluation(ABC):
    """
    Abstract base class for pre-retrieval evaluation.
    
    Defines the interface that all pre-retrieval evaluation implementations must follow.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the base pre-retrieval evaluation.
        
        Args:
            openai_api_key: OpenAI API key for LLM-based evaluation
        """
        self.openai_api_key = openai_api_key
        self.console = Console()
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate pre-retrieval strategy quality.
        
        Args:
            data: Input data dictionary specific to each strategy
            verbose: Whether to show detailed progress
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    @abstractmethod
    def load_sample_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load sample data from JSON file.
        
        Args:
            data_path: Path to the sample data file
            
        Returns:
            Dictionary with sample data
        """
        pass
    
    def validate_api_key(self) -> None:
        """
        Validate that OpenAI API key is provided when needed.
        
        Raises:
            ValueError: If API key is required but not provided
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for pre-retrieval evaluation")
    
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with loaded data
        """
        # Handle Hydra variable interpolation
        if '${hydra:runtime.cwd}' in file_path:
            import os
            file_path = file_path.replace('${hydra:runtime.cwd}', os.getcwd())
        
        data_file = Path(file_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.console.log(f"📂 Loaded data from: {data_file}", style="green")
        return data
    
    def calculate_average_scores(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average scores across all instances.
        
        Args:
            scores: Dictionary of scores per UUID
            
        Returns:
            Dictionary with average scores for each metric
        """
        metric_values = {}
        
        # Collect all values for each metric
        for uuid, metrics in scores.items():
            for metric_name, value in metrics.items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)
        
        # Calculate averages
        averages = {}
        for metric_name, values in metric_values.items():
            if values:
                averages[metric_name] = sum(values) / len(values)
        
        return averages