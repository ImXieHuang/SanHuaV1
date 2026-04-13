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
        """仿照以下内容实现的一个训练器函数
def static_trainer(self, inputs: List[List], outputs: List[List], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, rtn: RTN):
    original_tg = rtn.tg
    original_weights = rtn.weights
    try:
        rtn.tg = [[[0.0 for _ in j] for j in i] for i in rtn.tg]
        
        for ip, op in zip(inputs, outputs):
            print(f"Training {ip} -> {op}")
            
            Lam = 0.0
            Lambdacnt = 0
            for i in list(rtn.weights[0].values()):
                for j in list(i.values()):
                    Lam = add(Lam, lambdafunction(j))
                    Lambdacnt += 1
            for i in rtn.weights[1]:
                for j in i:
                    Lam = add(Lam, lambdafunction(j))
                    Lambdacnt += 1
            Lam = div(Lam, Lambdacnt) if Lambdacnt > 0 else 0.0
            
            wg = {}
            for start in list(rtn.weights[0].keys()):
                wg[start] = {}
                for end in list(rtn.weights[0][start].keys()):
                    def loss_with_reg(x):
                        return add(lossfunction(x, op), Lam)
                    wg[start][end] = self.weight_loss_derivative(start, end, ip, loss_with_reg, rtn)
            
            pg = {}
            for idx in range(len(rtn.weights[1])):
                pg[idx] = {}
                for jdx in range(len(rtn.weights[1][idx])):
                    def loss_with_reg(x):
                        return add(lossfunction(x, op), Lam)
                    pg[idx][jdx] = self.parameter_loss_derivative((idx, jdx), ip, loss_with_reg, rtn)
            
            for start in list(wg.keys()):
                for end in list(wg[start].keys()):
                    if uniform(0.0, 1.0) >= dropout:
                        delta = mul(r, wg[start][end])
                        if delta > maxdw:
                            delta = maxdw
                        if delta < -maxdw:
                            delta = -maxdw
                        rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], delta)
            
            for idx in list(pg.keys()):
                for jdx in list(pg[idx].keys()):
                    if uniform(0.0, 1.0) >= dropout:
                        delta = mul(r, pg[idx][jdx])
                        if delta > maxdw:
                            delta = maxdw
                        if delta < -maxdw:
                            delta = -maxdw
                        rtn.weights[1][idx][jdx] = sub(rtn.weights[1][idx][jdx], delta)

    except Exception as error:
        rtn.weights = original_weights
        rtn.tg = original_tg
        return error

    rtn.tg = original_tg
    return True
        """
        original_data = ccat.database
        try:
            Lam = 0.0
            Lambdacnt = 0
            for i in list(ccat.database.values()):
                for j,k in list(i.items()):
                    Lam = add(add(Lam, lambdafunction(j)), lambdafunction(k))
                    Lambdacnt += 1
            Lam = div(Lam, Lambdacnt) if Lambdacnt > 0 else 0.0

            for n_gram in [tokens[0:i] for i in range(1, len(tokens))]:
                print(f"Training on {n_gram} -> {n_gram[-1]}")
                
        except Exception as error:
            ccat.database = original_data
            return error

        return True

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

    t.trainer(tokens, t.cross_entropy, lambda x: x**2, 0.01, 0.1, 0.5, ccat)