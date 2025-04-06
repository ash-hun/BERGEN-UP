# ------------------------------------------------------------
# BERGEN-UP
# Copyright (c) 2025-present Ash-Hun.
# MIT license
# ------------------------------------------------------------
import warnings
from multiprocessing import set_start_method
import hydra
from hydra.core.config_store import ConfigStore
from config import Evaluation

# Hydra 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning, module="hydra")

cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=Evaluation)

@hydra.main(config_path = "conf", config_name="config", version_base="1.1")
def main(cfg: Evaluation):
    '''✨ Main function to run the BERGEN-UP ✨'''
    from modules.rag import RAG
    rag_module = RAG(config=cfg)
    rag_module.evaluate(verbose=True)
    print(f"Configuration: {cfg}")
    print("Hello from bergen-up!")


if __name__ == "__main__":
    set_start_method('spawn')
    main()