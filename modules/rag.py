from rich.console import Console

from config import Evaluation
from modules.chunking.evaluation import ChunkingEvaluation
from modules.retrieval.evaluation import RetrievalEvaluation
from modules.generation.evaluation import GenerationEvaluation
from modules.pre_retrieval.evaluation import PreRetrievalEvaluation
from modules.post_retrieval.evaluation import PostRetrievalEvaluation

class RAG:
    ''' 🥑 Orchestration of BERGEN-UP RAG pipeline 🥑 '''
    def __init__(self, config:Evaluation, **kwargs):
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
    
    def evaluate(self, verbose:bool=True):
        ''' Main function to run the BERGEN-UP RAG pipeline '''
        # Chunking Evaluation
        if hasattr(self.config.chunking, 'strategies') and self.config.chunking.strategies:
            # Title Logging
            self.console.log("Chunking Evaluation", style="bold yellow")
            chunking_evaluator = self._setup_chunking_evaluation()
            chunking_evaluator.run(verbose=verbose)
        else:
            # Exception Logging
            self.console.log("Chunking strategies not found in config", style="bold red")
        
        # Pre-retrieval Evaluation  
        if hasattr(self.config.pre_retrieval, 'strategies') and self.config.pre_retrieval.strategies:
            # Title Logging
            self.console.log("Pre-retrieval Evaluation", style="bold yellow")
            pre_retrieval_evaluator = self._setup_pre_retrieval_evaluation()
            pre_retrieval_evaluator.run(verbose=verbose)
        else:
            # Exception Logging
            self.console.log("Pre-retrieval strategies not found in config", style="bold red")

        # Retrieval Evaluation
        if hasattr(self.config.retrieval, 'strategies') and self.config.retrieval.strategies:
            # Title Logging
            self.console.log("Retrieval Evaluation", style="bold yellow")
            retrieval_evaluator = self._setup_retrieval_evaluation()
            retrieval_evaluator.run(verbose=verbose)
        else:
            # Exception Logging
            self.console.log("Retrieval strategies not found in config", style="bold red")

        # Post-retrieval Evaluation
        if hasattr(self.config.post_retrieval, 'strategies') and self.config.post_retrieval.strategies:
            # Title Logging
            self.console.log("Post-retrieval Evaluation", style="bold yellow")
            post_retrieval_evaluator = self._setup_post_retrieval_evaluation()
            post_retrieval_evaluator.run(verbose=verbose)
        else:
            # Exception Logging
            self.console.log("Post-retrieval strategies not found in config", style="bold red")

        # Generation Evaluation
        if hasattr(self.config.generation, 'strategies') and self.config.generation.strategies:
            # Title Logging
            self.console.log("Generation Evaluation", style="bold yellow")
            generation_evaluator = self._setup_generation_evaluation()
            generation_evaluator.run(verbose=verbose)
        else:
            # Exception Logging
            self.console.log("Generation strategies not found in config", style="bold red")