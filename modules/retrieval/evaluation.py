from typing import Optional, List

class RetrievalEvaluation:
    '''
    Retrieval Evaluation Module
    '''
    def __init__(
        self, 
        retrieval_strategy:List[dict],
        openai_api_key:Optional[str]=None
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.openai_api_key = openai_api_key
    
    def __str__(self) -> str:
        return "Retrieval Evaluation Module"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, verbose:bool=True):
        pass