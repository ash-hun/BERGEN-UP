# ------------------------------------------------------------
# BERGEN-UP
# Copyright (c) 2025-present Ash-Hun.
# MIT license
# ------------------------------------------------------------
import hydra
from multiprocessing import set_start_method
from hydra.core.config_store import ConfigStore
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="hydra") # Hydra 경고 메시지 필터링

import pprint
from modules.rag import RAG
from config import Evaluation

cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=Evaluation)

@hydra.main(config_path = "conf", config_name="config", version_base="1.1")
def main(cfg: Evaluation):
    '''✨ Main function to run the BERGEN-UP ✨'''
    pprint.pprint(cfg, indent=4)
    
    rag_module = RAG(config=cfg)
    rag_module.evaluate(verbose=False)
    
    print("Hello BERGEN-UP!")

if __name__ == "__main__":
    set_start_method('spawn')
    main()  # type: ignore # hydra.main 데코레이터가 자동으로 cfg를 주입합니다