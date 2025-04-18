import pandas as pd
from rich.console import Console
from typing import Optional, List
from chromadb.utils import embedding_functions

from modules.utils import rich_display_dataframe
from modules.chunking.utils import CustomEmbeddingFunction
from modules.chunking.chunker.semantic_chunker import SemanticChunker
from modules.chunking.chunker.recursive_token_chunker import RecursiveTokenChunker
from modules.chunking.chunker.fixed_token_chunker import TextSplitter, FixedTokenChunker
from modules.chunking.evaluation_framework.general_evaluation import GeneralEvaluation

class ChunkingEvaluation:
    ''' 🍋‍🟩 Chunking Evaluation Module 🍋‍🟩 '''
    def __init__(self, chunking_strategy:List[dict], openai_api_key:Optional[str]=None) -> None:
        self.questions_df_path = chunking_strategy[0]['question_set_path']
        self.corpora_id_path = chunking_strategy[1]['corpora_id_paths']
        if len(chunking_strategy) > 1:
            self.chunking_strategy = chunking_strategy[1:]
        else:
            raise ValueError("Chunking strategy must be included in the config file")
        self.openai_api_key = openai_api_key
        self.logger = Console()
        self.ef = None
    
    def __str__(self) -> str:
        return "Chunking Evaluation Module"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _set_recursive_token_chunker(self, **kwargs) -> RecursiveTokenChunker:
        return RecursiveTokenChunker(chunk_size=kwargs['chunk_size'], chunk_overlap=kwargs['chunk_overlap'])
    
    def _set_fixed_token_chunker(self, **kwargs) -> FixedTokenChunker:
        return FixedTokenChunker(chunk_size=kwargs['chunk_size'], chunk_overlap=kwargs['chunk_overlap'])
    
    def _set_semantic_chunker(self, **kwargs) -> SemanticChunker:
        if kwargs['mode'] == 'openai':
            self.ef = embedding_functions.OpenAIEmbeddingFunction(api_key=self.openai_api_key, model_name=kwargs['embedding_model'])
        elif kwargs['mode'] == 'custom':
            self.ef = CustomEmbeddingFunction(url=kwargs['custom_embedding_function'])
        return SemanticChunker(embedding_function=self.ef)
    
    def _get_chunking_strategy(self, verbose:bool=True) -> List[TextSplitter]:
        chunking_strategies = []
        for strategy in self.chunking_strategy:
            for strategy_name, params in strategy.items():
                if strategy_name == 'Recursive Token Chunking':
                    chunker = self._set_recursive_token_chunker(**params)
                    chunking_strategies.append(chunker)
                elif strategy_name == 'Fixed Token Chunking':
                    chunker = self._set_fixed_token_chunker(**params)
                    chunking_strategies.append(chunker)
                elif strategy_name == 'Semantic Chunking':
                    chunker = self._set_semantic_chunker(**params)
                    chunking_strategies.append(chunker)
        if verbose:
            print(chunking_strategies)
        return chunking_strategies
    
    def run(self, verbose:bool=False) -> pd.DataFrame:
        ''' Main function to run the chunking evaluation '''
        chunking_evaluator = GeneralEvaluation(
            questions_df_path=self.questions_df_path, 
            corpora_id_paths=self.corpora_id_path,
            verbose=verbose
        )
        # Run chunking evaluation
        results = []
        for chunker in self._get_chunking_strategy(verbose=verbose):
            result = chunking_evaluator.run(chunker, embedding_api_key=self.openai_api_key)
            del result['corpora_scores']  # Remove detailed scores for brevity
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else 0
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else 0
            result['chunker'] = (
                f"{chunker.__class__.__name__}_"
                f"{chunk_size}_{chunk_overlap}"
            )
            results.append(result)

        df = pd.DataFrame(results)
        rich_display_dataframe(df, title="Chunking Evaluation Results")
        return df