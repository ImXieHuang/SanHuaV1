import sys
from pathlib import Path
from typing import List, Callable
from random import uniform
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)
from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import Vector
import math

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
            sub_loss = loss(get_meaning_of_tokens_for_(ccat, tokens)[-1])
            ccat.database = original_data
            
            ccat.SoftInjection_to_(tokens[-1], bigQ, add(value, perturbation))
            add_loss = loss(get_meaning_of_tokens_for_(ccat, tokens)[-1])
            ccat.database = original_data
            
            gradient.components[i] = (add_loss - sub_loss) / (2 * self.dx)
        
        return gradient

if __name__ == '__main__':
    text = ['你好', '。', '我', '是', '三花', '。', '你', '叫', '什么', '名字', '？']
    c = NewCCATransformer(text)
    for i in range(9): c = FusionCCATransformer(c , NewCCATransformer(text))
    t = CCAT_Trainer(dx=1e-5)

    def l(x):
        def softmax(p):
            return math.exp(p)/(math.exp(i)+math.exp(j))
        
        hartley = []

        for i,j in zip(x.components, y.components):
            hartley.append(softmax(j) * math.log(softmax(i)) + (1-softmax(j)) * math.log(1-softmax(i)))

        return sum(hartley)

    cnt = 0
    while cnt < 50:
        cnt += 1
        print(f"{cnt = }")

        for i in [text[:j] for j in range(3, len(text))]:
            y = get_meaning_of_tokens_for_(c, i)[-1]
            if uniform(0.0, 1.0) >= 0.3:
                garadient = t.loss_gradient(i[:-1], l, c)
                print(f"token = {i[-1]}:\nbigQ = {y}, {garadient = }")
                c.SoftInjection_to_(i[-1], y, add(c.SoftQuery(i[-1], y), mul(garadient, 0.03)))
                c.SoftInjection_query_to_(i[-1], y)

    tokens = text[:2]
    print(tokens)
    while True:
        next_token = softmax_choice_next_token_for_(c, tokens)
        print(next_token)
        tokens.append(next_token)