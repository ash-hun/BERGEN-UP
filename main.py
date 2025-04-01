import hydra
from config import Evaluation
from hydra.core.config_store import ConfigStore

# Hydra 경고 메시지 필터링
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="hydra")

cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=Evaluation)

@hydra.main(config_path = "conf", config_name="config", version_base="1.1")
def main(cfg: Evaluation):
    '''
    ✨ Main function to run the BERGEN-UP ✨
    '''
    print("Hello from bergen-up!")


if __name__ == "__main__":
    main()