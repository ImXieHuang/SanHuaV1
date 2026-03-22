import sys
from pathlib import Path
from typing import Any
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Model.CCATransformer import CCATransformer, NewCCATransformer, get_meaning_of_tokens_for_, think_about_next_token_for_

class CCAT_Trainer:
    def __init__(self, dx: Any = None):
        self.dx = dx or 1e-5
        self.models_dir = self.get_models_dir()
    
    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, token: str, inputs: list, loss: callable, ccat: CCATransformer):
        pass

    def up_backward():
        pass

    def down_backward():
        pass

    def sub_backward():
        pass

    def ccat_trainer():
        pass
    
if __name__ == "__main__":
    t = CCAT_Trainer()