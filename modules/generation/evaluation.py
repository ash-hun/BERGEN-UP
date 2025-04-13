from typing import Optional, List

class GenerationEvaluation:
    '''
    Generation Evaluation Module
    '''
    def __init__(
        self, 
        generation_strategy:List[dict],
        openai_api_key:Optional[str]=None
    ) -> None:
        self.generation_strategy = generation_strategy
        self.openai_api_key = openai_api_key
    
    def __str__(self) -> str:
        return "Generation Evaluation Module"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, verbose:bool=True):
        pass