import sys
from pathlib import Path
from typing import Callable
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Model.CCATransformer import CCATransformer, NewCCATransformer, get_meaning_of_sentence_for_, think_about_next_token_for_
from Model.mathexpand import *
from Model.Vector import *

class CCAT_Trainer:
    def __init__(self, dx: float = None):
        self.dx = dx or 1e-5
        self.models_dir = self.get_models_dir()
    
    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, tokens: str, loss: Callable, ccat: CCATransformer):
        original_data = ccat.database
        bigQ = get_meaning_of_sentence_for_(ccat, tokens[:-1])
        data = get_meaning_of_sentence_for_(ccat, tokens)

        ccat.SoftInjection_to_(tokens[-1], bigQ, add(data, [self.dx for _ in data]))
        add_loss = loss(get_meaning_of_sentence_for_(ccat, tokens))
        ccat.database = original_data
        
        ccat.SoftInjection_to_(tokens[-1], bigQ, sub(data, [self.dx for _ in data]))
        sub_loss = loss(get_meaning_of_sentence_for_(ccat, tokens))
        ccat.database = original_data

        return div(div(sub(add_loss, sub_loss), 2), self.dx)
    
    def ccat_trainer():
        pass
    
if __name__ == "__main__":
    t = CCAT_Trainer()