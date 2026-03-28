import sys
from pathlib import Path
from typing import List, Callable

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import *
from Model.fraction import fraction
import Model.mathexpand as mexp
from Model.Vector.vector import Vector

class CCAT_Trainer:
    def __init__(self, dx: fraction = None):
        self.dx = dx or fraction(1, 1000)
        self.models_dir = self.get_models_dir()
    
    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, tokens: List[str], loss: Callable, ccat: CCATransformer):
        pass

if __name__ == "__main__":
    text = ["你好", "我", "是", "三花"]
    
    c = FusionCCATransformer(NewCCATransformer(text), NewCCATransformer(text))
    t = CCAT_Trainer(dx=fraction(1, 100))
    
    for i in range(1, len(text)):
        q = get_meaning_of_sentence_for_(c, text[:i+1])
        
        def loss_fn(x):
            diff = mexp.sub(x, q)
            squared_sum = fraction(0, 1)
            for comp in diff.components:
                if isinstance(comp, fraction):
                    squared_sum = mexp.add(squared_sum, mexp.mul(comp, comp))
                else:
                    f_comp = fraction(comp, 1)
                    squared_sum = mexp.add(squared_sum, mexp.mul(f_comp, f_comp))
            return Vector([squared_sum for _ in range(c.dim)])
        
        gradient = t.loss_gradient(text[:i+1], loss_fn, c)
        
        lr = fraction(1, 100)
        for token in set(text[:i+1]):
            current_vec = get_meaning_of_sentence_for_(c, [token])
            update = mexp.mul(gradient.components, [lr for _ in range(c.dim)])
            new_vec = mexp.sub(current_vec, Vector(update))
            for query in c.database[token]:
                c.database[token][query] = new_vec