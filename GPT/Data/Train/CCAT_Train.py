import sys
import copy
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
            original_data = copy.deepcopy(ccat.database)
            bigQ = get_meaning_of_sentence_for_(ccat, tokens[:-1])
            
            perturbation = Vector([0.0] * dim)
            perturbation.components[i] = self.dx
            
            value = ccat.SoftQuery(tokens[-1], bigQ)

            ccat.SoftInjection_to_(tokens[-1], bigQ, sub(value, perturbation))
            sub_loss = loss(tokens, ccat)
            ccat.database = copy.deepcopy(original_data)
            
            ccat.SoftInjection_to_(tokens[-1], bigQ, add(value, perturbation))
            add_loss = loss(tokens, ccat)
            ccat.database = copy.deepcopy(original_data)
            
            gradient.components[i] = (add_loss - sub_loss) / (2 * self.dx)
        
        return gradient
    
    def trainer(self, tokens: List[str], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, ccat: CCATransformer):
        original_data = copy.deepcopy(ccat.database)
        try:
            def loss(token, ccat) -> float:
                Lam = 0.0
                Lambdacnt = 0
                for i in list(ccat.database.values()):
                    for j,k in list(i.items()):
                        Lam = add(add(Lam, lambdafunction(j)), lambdafunction(k))
                        Lambdacnt += 1
                Lam = sum(Lam.components)/Lambdacnt if Lambdacnt > 0 else 0.0

                return lossfunction(token, ccat) + Lam

            injection = {}

            for n_gram in [tokens[:i] for i in range(2, len(tokens))]:
                if rand.random() < dropout:
                    continue
                print(f"Training on {n_gram[:-1]} -> {n_gram[-1]}")
                gradient = self.loss_gradient(n_gram, loss, ccat)
                bigQ = get_meaning_of_sentence_for_(ccat, n_gram[:-1])
                injection[tuple(n_gram)] = (gradient, bigQ)
                print("Probabilities:")
                for token, prob in zip(*softmax_choice_next_probability_for_(ccat, n_gram[:-1])):
                    print(f"  {token}: {prob:.4f}")
                print(f"Loss: {loss(n_gram, ccat):.4f}\n")

            for token, data in injection.items():
                token = list(token)
                grad = data[0]
                bigQ = data[1]

                if grad.magnitude() > maxdw:
                    grad = grad.normalize() * maxdw
                value = ccat.SoftQuery(token[-1], bigQ)

                ccat.SoftInjection_to_(token[-1], bigQ, sub(value, grad))
                ccat.SoftInjection_query_to_(token[-1], bigQ)

        except Exception as error:
            ccat.database = copy.deepcopy(original_data)
            return error

        return True

    def cross_entropy(self, tokens: List[str], actuality: str, ccat: CCATransformer) -> float:
        p = softmax_choice_next_probability_for_(ccat, tokens)
        if actuality not in p[0]:
            return float('inf')
        return -log2(p[1][p[0].index(actuality)])

if __name__ == "__main__":
    from Data.splitToken import ChineseTokenizer
    from Model.mathexpand import iterate
    from time import time
    import sys

    log_file = Path(__file__).parent / "training_log.txt"
    sys.stdout = open(log_file, "w")

    start_time = time()

    print("Loading and preparing data...")

    text = ["""这是一个测试序列""",
            """是一个测试序列"""]

    ct = ChineseTokenizer()
    ct.load_model("model")

    Texts = [["<START>"] + ct.tokenize(i) + ["<END>"]*2 for i in text]

    print(f"Data loaded and prepared: {Texts}")

    print(f"Data preparation completed in {time() - start_time:.2f} seconds.\n")

    sys.stdout.flush()

    print()

    start_time = time()
    
    print("Preparing CCAT model...")

    ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(list({i for j in Texts for i in j}), dim = 2) for _ in range(4)])

    ccat.temperature = 1.0

    print(f"CCAT model preparation completed in {time() - start_time:.2f} seconds.\n")

    print("Starting training...")

    t = CCAT_Trainer()
    r = 0.75

    for cnt in range(500):
        start_time = time()

        print(f"Epoch {cnt + 1}")

        for Text in Texts:  
            print(f"Training on {Text} are {t.trainer(Text, lambda tokens, ccat: t.cross_entropy(tokens, Text[Text.index(tokens[-1]) + 1] if len(tokens) > 1 else 0.0 + sum([i[1]**4 if i[0] in ['<START>'] else 0.0 for i in softmax_choice_next_probability_for_(ccat, tokens)]), ccat), lambda x: 0.01 * x**2, r, 2.5, 0.1, ccat)}")
            print()

        print(f"Epoch {cnt + 1} completed in {time() - start_time:.2f} seconds.\n")

        print("test:")

        for i in range(len(Texts)):
            print(f"Sequence: {''.join(Texts[i])}")
            for j in range(len(Texts[i]) - 1):
                print(f"{'……' if j >= 1 else '  '} {Texts[i][j]} -> {Texts[i][j + 1]}:   {softmax_choice_next_token_for_(ccat, Texts[i][:j])}")
            print()

        print()

        sys.stdout.flush()

    print("Training completed.")

    print()

    print("Testing the model 10 times...")

    for i in range(10):
        tokens = ["这"]
        while tokens[-1] != "<END>" and len(tokens) < 20:
            next_token = softmax_choice_next_token_for_(ccat, tokens)
            tokens.append(next_token)
        print(f"Generated sequence {i + 1}: {''.join(tokens)}")