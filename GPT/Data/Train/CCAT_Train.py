import sys
from pathlib import Path
from typing import List, Callable
import random as rand
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)
from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import Vector
from math import log2

class CCAT_Trainer:
    def __init__(self, dx: float=None):
        self.dx = dx or 0.001
        self.models_dir = self.get_models_dir()

    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, tokens: List[str], loss: Callable, ccat: CCATransformer) -> Vector:
        dim = ccat.dim
        gradient = Vector([0.0] * dim)
        
        for i in range(dim):
            original_data = ccat.database
            bigQ = get_meaning_of_sentence_for_(ccat, tokens[:-1])
            
            perturbation = Vector([0.0] * dim)
            perturbation.components[i] = self.dx
            
            value = ccat.SoftQuery(tokens[-1], bigQ)

            ccat.SoftInjection_to_(tokens[-1], bigQ, sub(value, perturbation))
            sub_loss = loss(tokens, ccat)
            ccat.database = original_data
            
            ccat.SoftInjection_to_(tokens[-1], bigQ, add(value, perturbation))
            add_loss = loss(tokens, ccat)
            ccat.database = original_data
            
            gradient.components[i] = (add_loss - sub_loss) / (2 * self.dx)
        
        return gradient
    
    def trainer(self, tokens: List[str], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, ccat: CCATransformer):
        pass

    def cross_entropy(self, tokens: List[str], actuality: str):
        p = softmax_choice_next_probability_for_(ccat, tokens)
        return -log2(p[1][p[0].index(actuality)])

if __name__ == "__main__":
    t =  CCAT_Trainer()

    data = ["a","b","c"]

    ccat = NewCCATransformer(data)

    tokens = rand.choices(data, k=5)
    actuality = rand.choice(data)

    h = t.cross_entropy(tokens, actuality)

    print(f"debug:\n{tokens = }, {actuality = }, {h = }")