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
        original_data = ccat.database
        try:
            Lam = 0.0
            Lambdacnt = 0
            for i in list(ccat.database.values()):
                for j,k in list(i.items()):
                    Lam = add(add(Lam, lambdafunction(j)), lambdafunction(k))
                    Lambdacnt += 1
            Lam = sum(Lam.components)/Lambdacnt if Lambdacnt > 0 else 0.0

            for n_gram in [tokens[0:i] for i in range(2, len(tokens))]:
                print(f"Training on {n_gram[0:-1]} -> {n_gram[-1]}")
                def loss_with_reg(tokens, ccat):
                    return add(lossfunction(tokens, ccat), Lam)
                gradient = self.loss_gradient(n_gram, loss_with_reg, ccat)
                for i in range(ccat.dim):
                    if rand.uniform(0.0, 1.0) >= dropout:
                        delta = r * gradient.components[i]
                        if delta > maxdw:
                            delta = maxdw
                        if delta < -maxdw:
                            delta = -maxdw
                        ccat.SoftInjection_to_(n_gram[-1], get_meaning_of_sentence_for_(ccat, n_gram[:-1]), delta)
                        ccat.SoftInjection_query_to_(n_gram[-1], get_meaning_of_sentence_for_(ccat, n_gram[:-1]))
                
        except Exception as error:
            ccat.database = original_data
            return error

        return True

    def cross_entropy(self, tokens: List[str], actuality: str, ccat: CCATransformer) -> float:
        p = softmax_choice_next_probability_for_(ccat, tokens)
        return -log2(p[1][p[0].index(actuality)])

if __name__ == "__main__":
    from Data.splitToken import ChineseTokenizer
    from Model.mathexpand import iterate


    text = """大家好，我是三花V1"""

    ct = ChineseTokenizer()
    ct.load_model("model")

    Texts = ct.tokenize(text) + ['<end>']

    ccat = iterate(FusionCCATransformer, [NewCCATransformer(Texts, 8) for _ in range(4)])

    t = CCAT_Trainer()
    for cnt in range(100):
        print(f"Epoch {cnt + 1}")

        print(t.trainer(Texts, lambda tokens, ccat: t.cross_entropy(tokens, Texts[Texts.index(tokens[-1]) + 1] if len(tokens) > 1 else None, ccat), lambda x: 0.05 * x**2, 0.1, 0.5, 0.2, ccat))
        
        print("Model data:")
        for i in ccat.database:
            print(f"{i}:")
            for j in ccat.database[i]:
                print(f"  {j} : {ccat.database[i][j]}")
        
        print("test:")

        for i in range(len(Texts) - 1):
            print(f"{Texts[i]} -> {Texts[i + 1]}:   {softmax_choice_next_token_for_(ccat, Texts[:i + 1])}")
            print(f"loss: {t.cross_entropy(Texts[:i + 1], Texts[i + 1], ccat)}")
            print()
    
    print("Training completed.")