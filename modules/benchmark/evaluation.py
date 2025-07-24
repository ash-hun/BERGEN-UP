"""
Benchmark Evaluation Module for BERGEN-UP
Evaluates LLM performance using various benchmarks
"""
import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Any
from rich.console import Console
from pathlib import Path
import pandas as pd

from modules.utils import rich_display_dataframe
from modules.benchmark.evaluation_framework import NIAHEvaluation


class BenchmarkEvaluation:
    """
    🎯 Benchmark Evaluation Module 🎯
    
    Evaluates LLM endpoints using standardized benchmarks:
    - NIAH (Needle In A Haystack): Long-context comprehension
    - More benchmarks can be added (BEIR, etc.)
    """
    
    def __init__(self, 
                 benchmark_strategy: List[dict],
                 openai_api_key: Optional[str] = None) -> None:
        """
        Initialize Benchmark Evaluation
        
        Args:
            benchmark_strategy: List of benchmark configurations
            openai_api_key: Optional OpenAI API key
        """
        self.benchmark_strategy = benchmark_strategy
        self.openai_api_key = openai_api_key
        self.console = Console()
        
        # Parse configuration
        self._parse_strategies()
    
    def _run_async_benchmark(self, benchmark_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to run async benchmark in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if benchmark_name == 'niah':
                niah_eval = self._setup_niah_evaluation(config)
                results = loop.run_until_complete(niah_eval.run())
                return results
            return {}
        finally:
            # Clean up any remaining tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        
    def __str__(self) -> str:
        return "🎯 Benchmark Evaluation Module 🎯"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _parse_strategies(self) -> None:
        """Parse benchmark strategies from configuration"""
        self.benchmarks_to_run = []
        self.llm_endpoint = None
        self.needle_config_path = None
        
        for strategy in self.benchmark_strategy:
            if 'llm_endpoint' in strategy:
                self.llm_endpoint = strategy['llm_endpoint']
            elif 'needle_config_path' in strategy:
                self.needle_config_path = strategy['needle_config_path']
            elif 'NIAH' in strategy:
                niah_config = strategy['NIAH']
                self.benchmarks_to_run.append(('niah', niah_config))
            # Add more benchmark types here as needed
        
        if not self.llm_endpoint:
            raise ValueError("LLM endpoint must be provided in benchmark configuration")
    
    def _setup_niah_evaluation(self, config: Dict[str, Any]) -> NIAHEvaluation:
        """Setup NIAH evaluation with configuration"""
        return NIAHEvaluation(
            llm_endpoint=self.llm_endpoint,
            needle_config_path=self.needle_config_path,
            context_lengths=config.get('context_lengths'),
            document_depth_percents=config.get('document_depth_percents'),
            haystack_dir=config.get('haystack_dir'),
            num_samples_per_test=config.get('num_samples_per_test', 5),
            save_results=config.get('save_results', True),
            save_contexts=config.get('save_contexts', False),
            test_cases=config.get('test_cases'),  # Add test_cases parameter
            openai_api_key=self.openai_api_key
        )
    
    def run(self, verbose: bool = True):
        """
        Run benchmark evaluation
        
        Args:
            verbose: Whether to display detailed output
            
        Returns:
            Dict with detailed benchmark results or DataFrame for backward compatibility
        """
        self.console.log("Starting Benchmark Evaluation", style="bold yellow")
        
        # Use asyncio to run async benchmarks
        all_results = []
        detailed_results = None
        
        try:
            # Check if we're already in an async context (like FastAPI)
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context, run in a separate thread
                for benchmark_name, config in self.benchmarks_to_run:
                    if benchmark_name == 'niah':
                        self.console.log(f"Running NIAH benchmark", style="cyan")
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(self._run_async_benchmark, benchmark_name, config)
                            results = future.result()
                        
                        # Store detailed results for API
                        detailed_results = results
                        
                        # Extract key metrics for summary
                        if 'analysis' in results:
                            for test_name, analysis in results['analysis'].items():
                                all_results.append({
                                    'benchmark': 'NIAH',
                                    'test_case': test_name,
                                    'overall_accuracy': analysis['overall_accuracy'],
                                    'llm_endpoint': self.llm_endpoint
                                })
                        
                        if verbose:
                            self.console.log(f"NIAH benchmark completed", style="green")
                    
                    # Add more benchmark types here as they are implemented
                        
            except RuntimeError:
                # No current loop, safe to create a new one (CLI context)
                for benchmark_name, config in self.benchmarks_to_run:
                    if benchmark_name == 'niah':
                        self.console.log(f"Running NIAH benchmark", style="cyan")
                        
                        results = self._run_async_benchmark(benchmark_name, config)
                        
                        # Store detailed results for API
                        detailed_results = results
                        
                        # Extract key metrics for summary
                        if 'analysis' in results:
                            for test_name, analysis in results['analysis'].items():
                                all_results.append({
                                    'benchmark': 'NIAH',
                                    'test_case': test_name,
                                    'overall_accuracy': analysis['overall_accuracy'],
                                    'llm_endpoint': self.llm_endpoint
                                })
                        
                        if verbose:
                            self.console.log(f"NIAH benchmark completed", style="green")
                    
                    # Add more benchmark types here as they are implemented
                        
        except Exception as e:
            self.console.print(f"[red]Error in benchmark evaluation: {e}[/red]")
        
        # Store detailed results as instance attribute for API access
        self.detailed_results = detailed_results
        
        # Return detailed results if available (for API), otherwise summary DataFrame (for CLI)
        if detailed_results:
            return detailed_results
        elif all_results:
            df = pd.DataFrame(all_results)
            if verbose:
                rich_display_dataframe(df, title="Benchmark Evaluation Summary")
            return df
        else:
            self.console.log("No benchmark results generated", style="yellow")
            return pd.DataFrame()