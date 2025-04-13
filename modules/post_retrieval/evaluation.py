from typing import Optional, List

class PostRetrievalEvaluation:
    '''
    Post-retrieval Evaluation Module
    '''
    def __init__(
        self, 
        post_retrieval_strategy:List[dict],
        openai_api_key:Optional[str]=None
    ) -> None:
        self.post_retrieval_strategy = post_retrieval_strategy
        self.openai_api_key = openai_api_key
    
    def __str__(self) -> str:
        return "Post-retrieval Evaluation Module"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def run(self, verbose:bool=True):
        ''' Main function to run the post-retrieval evaluation '''
        pass