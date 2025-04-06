import os
from typing import Optional
from importlib import resources
from modules.chunking.evaluation_framework.base_evaluation import BaseEvaluation

class GeneralEvaluation(BaseEvaluation):
    def __init__(
            self, 
            questions_df_path:Optional[str]=None,
            chroma_db_path:Optional[str]=None, 
            corpora_id_paths:Optional[dict]=None,
            verbose:bool=False
        ):
        with resources.as_file(resources.files('modules.chunking.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
            if questions_df_path is None:
                questions_df_path = os.path.join(general_benchmark_path, 'questions_df_chatlogs.csv')
            
            if corpora_id_paths is None:
                corpora_id_paths = {
                    'chatlogs': os.path.join(general_benchmark_path, 'corpora', 'chatlogs.md')
                }

            if verbose:
                print(f"▶ Questions_df_path : {str(questions_df_path)}")
                print(f"▶ Chroma_db_path : {str(chroma_db_path)}")
                print(f"▶ Corpora_id_paths : {corpora_id_paths}")
            
            super().__init__(questions_csv_path=questions_df_path, chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)
            self.is_general = True
