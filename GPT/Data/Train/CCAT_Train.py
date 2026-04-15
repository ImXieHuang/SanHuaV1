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
                injection = []
                for i in range(ccat.dim):
                    delta = 0.0
                    if rand.uniform(0.0, 1.0) >= dropout:
                        delta = -r * gradient.components[i]
                        if delta > maxdw:
                            delta = maxdw
                        if delta < -maxdw:
                            delta = -maxdw
                    injection.append(delta)
                injection = ccat.SoftQuery(n_gram[-1], get_meaning_of_sentence_for_(ccat, n_gram[:-1])) + Vector(injection)

                ccat.SoftInjection_to_(n_gram[-1], get_meaning_of_sentence_for_(ccat, n_gram[:-1]), injection)
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


    text = """我常想在纷扰中寻出一点闲静来，然而委实不容易。目前是这么离奇，心里是这么芜杂。一个人做到只剩了回忆的时候，生涯大概总要算是无聊了罢，但有时竟会连回忆也没有。中国的做文章有轨范，世事也仍然是螺旋。前几天我离开中山大学的时候，便想起四个月以前的离开厦门大学；听到飞机在头上鸣叫，竟记得了一年前在北京城上日日旋绕的飞机。我那时还做了一篇短文，叫做《一觉》。现在是，连这“一觉”也没有了。
广州的天气热得真早，夕阳从西窗射入，逼得人只能勉强穿一件单衣。书桌上的一盆“水横枝”，是我先前没有见过的：就是一段树，只要浸在水中，枝叶便青葱得可爱。看看绿叶，编编旧稿，总算也在做一点事。做着这等事，真是虽生之日，犹死之年，很可以驱除炎热的。
前天，已将《野草》编定了；这回便轮到陆续载在《莽原》上的《旧事重提》，我还替他改了一个名称：《朝花夕拾》。带露折花，色香自然要好得多，但是我不能够。便是现在心目中的离奇和芜杂，我也还不能使他即刻幻化，转成离奇和芜杂的文章。或者，他日仰看流云时，会在我的眼前一闪烁罢。
我有一时，曾经屡次忆起儿时在故乡所吃的蔬果：菱角、罗汉豆、茭白、香瓜。凡这些，都是极其鲜美可口的；都曾是使我思乡的蛊惑。后来，我在久别之后尝到了，也不过如此；惟独在记忆上，还有旧来的意味存留。他们也许要哄骗我一生，使我时时反顾。
这十篇就是从记忆中抄出来的，与实际容或有些不同，然而我现在只记得是这样。文体大概很杂乱，因为是或作或辍，经了九个月之多。环境也不一：前两篇写于北京寓所的东壁下；中三篇是流离中所作，地方是医院和木匠房；后五篇却在厦门大学的图书馆的楼上，已经是被学者们挤出集团之后了。
一九二七年五月一日，鲁迅于广州白云楼记。"""

    ct = ChineseTokenizer()
    ct.load_model("model")

    Texts = ct.tokenize(text) + ['<end>']

    ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(Texts, 8) for _ in range(4)])

    ccat.temperature = 0.25

    t = CCAT_Trainer()
    for cnt in range(100):
        print(f"Epoch {cnt + 1}")

        print(t.trainer(Texts, lambda tokens, ccat: t.cross_entropy(tokens, Texts[Texts.index(tokens[-1]) + 1] if len(tokens) > 1 else None, ccat), lambda x: 0.05 * x**2, 0.5, 1.0, 0.2, ccat))
        
        print("test:")

        for i in range(len(Texts) - 1):
            print(f"{Texts[i]} -> {Texts[i + 1]}:   {softmax_choice_next_token_for_(ccat, Texts[:i + 1])}")
            print(f"loss: {t.cross_entropy(Texts[:i + 1], Texts[i + 1], ccat)}")
            print()
    
    print("Training completed.")