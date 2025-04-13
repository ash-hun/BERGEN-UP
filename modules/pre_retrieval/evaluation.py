from typing import Optional, List

class PreRetrievalEvaluation:
    '''
    Pre-retrieval Evaluation Module
    '''
    def __init__(
        self, 
        pre_retrieval_strategy:List[dict],
        openai_api_key:Optional[str]=None
    ) -> None:
        self.pre_retrieval_strategy = pre_retrieval_strategy
        self.openai_api_key = openai_api_key
    
    def __str__(self) -> str:
        return "Pre-retrieval Evaluation Module"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, verbose:bool=True):
        ''' Main function to run the pre-retrieval evaluation '''
        pass