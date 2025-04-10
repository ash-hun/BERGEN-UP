from config import Evaluation
from modules.chunking.evaluation import ChunkingEvaluation

class RAG:
    def __init__(self, config:Evaluation, **kwargs):
        self.config = config
    
    def __str__(self):
        return f"RAG(config={self.config})"
    
    def __repr__(self):
        return self.__str__()
    
    def _setup_chunking_evaluation(self) -> ChunkingEvaluation:
        return ChunkingEvaluation(
            chunking_strategy=self.config.chunking.strategies,
            openai_api_key=self.config.common.OPENAI_API_KEY
        )
    
    def evaluate(self, verbose:bool=True):
        ch_e = self._setup_chunking_evaluation()
        ch_e.run(verbose=verbose)