from rich.console import Console

from config import Evaluation
from modules.chunking.evaluation import ChunkingEvaluation
from modules.retrieval.evaluation import RetrievalEvaluation
from modules.generation.evaluation import GenerationEvaluation
from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
from modules.post_retrieval.evaluation import PostRetrievalEvaluation
from modules.benchmark.evaluation import BenchmarkEvaluation
from modules.benchmark.function_chat.evaluation import FunctionChatEvaluation

class RAG:
    ''' 🥑 Orchestration of BERGEN-UP RAG pipeline 🥑 '''
    def __init__(self, config:Evaluation):
        self.config = config
        self.console = Console()
    
    def __str__(self):
        return f"RAG(config={self.config})"
    
    def __repr__(self):
        return self.__str__()
    
    def _setup_chunking_evaluation(self) -> ChunkingEvaluation:
        return ChunkingEvaluation(
            chunking_strategy=self.config.chunking.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_pre_retrieval_evaluation(self) -> PreRetrievalEvaluation:
        return PreRetrievalEvaluation(
            pre_retrieval_strategy=self.config.pre_retrieval.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_retrieval_evaluation(self) -> RetrievalEvaluation:
        return RetrievalEvaluation(
            retrieval_strategy=self.config.retrieval.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_post_retrieval_evaluation(self) -> PostRetrievalEvaluation:
        return PostRetrievalEvaluation(
            post_retrieval_strategy=self.config.post_retrieval.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_generation_evaluation(self) -> GenerationEvaluation:
        return GenerationEvaluation(
            generation_strategy=self.config.generation.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_benchmark_evaluation(self) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(
            benchmark_strategy=self.config.benchmark.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def _setup_function_chat_evaluation(self) -> FunctionChatEvaluation:
        # Convert config to dictionary format expected by FunctionChatEvaluation
        function_chat_config = {
            'output_dir': './outputs'
        }
        
        # Extract function_chat specific config from strategies
        if hasattr(self.config.function_chat, 'strategies') and self.config.function_chat.strategies:
            for strategy in self.config.function_chat.strategies:
                # OmegaConf DictConfig should be treated like dict
                if hasattr(strategy, 'items'):
                    for key, value in strategy.items():
                        # Handle special cases for API key interpolation
                        if key == 'llm_api_key' and isinstance(value, str) and '${common.OPENAI_API_KEY}' in value:
                            function_chat_config[key] = self.config.common.OPENAI_API_KEY
                        else:
                            function_chat_config[key] = value
        
        
        # Ensure data_path is properly resolved
        if 'data_path' in function_chat_config:
            import os
            from hydra.core.hydra_config import HydraConfig
            
            data_path = function_chat_config['data_path']
            # Handle Hydra variable interpolation
            if '${hydra:runtime.cwd}' in data_path:
                try:
                    # Get the original working directory before Hydra changed it
                    hydra_cfg = HydraConfig.get()
                    original_cwd = hydra_cfg.runtime.cwd
                    data_path = data_path.replace('${hydra:runtime.cwd}', original_cwd)
                except:
                    # Fallback: use the parent directory of the current file as project root
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_path = data_path.replace('${hydra:runtime.cwd}', project_root)
            
            # Ensure absolute path
            if not os.path.isabs(data_path):
                try:
                    hydra_cfg = HydraConfig.get()
                    original_cwd = hydra_cfg.runtime.cwd
                    data_path = os.path.join(original_cwd, data_path)
                except:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_path = os.path.join(project_root, data_path)
            
            function_chat_config['data_path'] = data_path
        
        return FunctionChatEvaluation(function_chat_config)
    
    def evaluate(self, verbose:bool=True) -> None:
        ''' [ Main function to run the BERGEN-UP RAG pipeline ] '''
        # Chunking Evaluation
        try:
            if hasattr(self.config.chunking, 'strategies') and self.config.chunking.strategies:
                # Title Logging
                self.console.log("Chunking Evaluation", style="bold yellow")
                chunking_evaluator = self._setup_chunking_evaluation()
                chunking_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Chunking strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Chunking strategies not found in config", style="bold red")
        
        # Pre-retrieval Evaluation  
        try:
            if hasattr(self.config.pre_retrieval, 'strategies') and self.config.pre_retrieval.strategies:
                # Title Logging
                self.console.log("Pre-retrieval Evaluation", style="bold yellow")
                pre_retrieval_evaluator = self._setup_pre_retrieval_evaluation()
                pre_retrieval_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Pre-retrieval strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Pre-retrieval strategies not found in config", style="bold red")

        # Retrieval Evaluation
        try:
            if hasattr(self.config.retrieval, 'strategies') and self.config.retrieval.strategies:
                # Title Logging
                self.console.log("Retrieval Evaluation", style="bold yellow")
                retrieval_evaluator = self._setup_retrieval_evaluation()
                retrieval_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Retrieval strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Retrieval strategies not found in config", style="bold red")

        # Post-retrieval Evaluation
        try:
            if hasattr(self.config.post_retrieval, 'strategies') and self.config.post_retrieval.strategies:
                # Title Logging
                self.console.log("Post-retrieval Evaluation", style="bold yellow")
                post_retrieval_evaluator = self._setup_post_retrieval_evaluation()
                post_retrieval_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Post-retrieval strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Post-retrieval strategies not found in config", style="bold red")

        # Generation Evaluation
        try:
            if hasattr(self.config.generation, 'strategies') and self.config.generation.strategies:
                # Title Logging
                self.console.log("Generation Evaluation", style="bold yellow")
                generation_evaluator = self._setup_generation_evaluation()
                generation_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Generation strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Generation strategies not found in config", style="bold red")
        
        # Benchmark Evaluation
        try:
            if hasattr(self.config, 'benchmark') and hasattr(self.config.benchmark, 'strategies') and self.config.benchmark.strategies:
                # Title Logging
                self.console.log("Benchmark Evaluation", style="bold yellow")
                benchmark_evaluator = self._setup_benchmark_evaluation()
                benchmark_evaluator.run(verbose=verbose)
            else:
                # Exception Logging
                self.console.log("Benchmark strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("Benchmark strategies not found in config", style="bold red")
        
        # FunctionChat Evaluation
        try:
            if hasattr(self.config, 'function_chat') and hasattr(self.config.function_chat, 'strategies') and self.config.function_chat.strategies:
                # Title Logging
                self.console.log("FunctionChat Evaluation", style="bold yellow")
                function_chat_evaluator = self._setup_function_chat_evaluation()
                function_chat_evaluator.run_evaluation()
            else:
                # Exception Logging
                self.console.log("FunctionChat strategies not found in config", style="bold red")
        except AttributeError:
            # Exception Logging
            self.console.log("FunctionChat strategies not found in config", style="bold red")