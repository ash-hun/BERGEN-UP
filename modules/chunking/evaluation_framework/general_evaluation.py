import os
from typing import Optional
from rich.console import Console
from modules.chunking.evaluation_framework.base_evaluation import BaseEvaluation

class GeneralEvaluation(BaseEvaluation):
    def __init__(
            self, 
            questions_df_path:Optional[str]=None,
            chroma_db_path:Optional[str]=None, 
            corpora_id_paths:Optional[dict]=None,
            verbose:bool=False
        ):
        self.console = Console()
        
        # 프로젝트 루트 디렉토리 경로 설정 (pipeline.py가 있는 디렉토리)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        data_dir = os.path.join(project_root, "data")
        
        if questions_df_path is None:
            questions_df_path = os.path.join(data_dir, "chunking", "question_set", "questions_df_chatlogs.csv")
            if not os.path.exists(questions_df_path):
                self.console.log(f"❌ File not found: {questions_df_path}", style="bold red")
                self.console.log(f"Project root: {project_root}", style="bold yellow")
                raise FileNotFoundError(f"Questions file not found at: {questions_df_path}")
        
        if corpora_id_paths is None:
            chatlogs_path = os.path.join(data_dir, "chunking", "corpora", "chatlogs.md")
            if not os.path.exists(chatlogs_path):
                self.console.log(f"❌ File not found: {chatlogs_path}", style="bold red")
                self.console.log(f"Project root: {project_root}", style="bold yellow")
                raise FileNotFoundError(f"Chatlogs file not found at: {chatlogs_path}")
            corpora_id_paths = {
                'chatlogs': chatlogs_path
            }

        if verbose:
            self.console.log(f"▶ Project root: {project_root}", style="bold blue")
            self.console.log(f"▶ Data directory: {data_dir}", style="bold blue")
            self.console.log(f"▶ Questions_df_path: {questions_df_path}", style="bold blue")
            self.console.log(f"▶ Chroma_db_path: {str(chroma_db_path)}", style="bold blue")
            self.console.log(f"▶ Corpora_id_paths: {corpora_id_paths}", style="bold blue")
        
        super().__init__(questions_csv_path=questions_df_path, chroma_db_path=chroma_db_path, corpora_id_paths=corpora_id_paths)
        self.is_general = True
