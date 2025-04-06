from typing import Optional
from importlib import resources
from modules.chunking.evaluation_framework.base_evaluation import BaseEvaluation

class GeneralEvaluation(BaseEvaluation):
    def __init__(
            self, 
            questions_df_path:Optional[str]='./evaluation_framework/general_evaluation_data/questions_df_chatlogs.csv', 
            chroma_db_path:Optional[str]=None, 
            corpora_id_paths:Optional[dict]={'chatlogs': './evaluation_framework/general_evaluation_data/corpora/chatlogs.md'}, 
            verbose:bool=False
        ):
        # with resources.as_file(resources.files('chunking_evaluation.evaluation_framework') / 'general_evaluation_data') as general_benchmark_path:
        #     self.general_benchmark_path = general_benchmark_path
        #     questions_df_path = self.general_benchmark_path / 'questions_df.csv'

        #     corpora_folder_path = self.general_benchmark_path / 'corpora'
        #     corpora_filenames = [f for f in corpora_folder_path.iterdir() if f.is_file()]

        #     corpora_id_paths = {
        #         f.stem: str(f) for f in corpora_filenames
        #     }

            if verbose:
                print(f"▶ Questions_df_path : {str(questions_df_path)}")
                print(f"▶ Chroma_db_path : {str(chroma_db_path)}")
                print(f"▶ Corpora_id_paths : {corpora_id_paths}")
            
            super().__init__(questions_csv_path=questions_df_path, chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)
            self.is_general = True
