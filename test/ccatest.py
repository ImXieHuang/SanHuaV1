import sys
from pathlib import Path
from random import uniform

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from GPT.Model.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from GPT.Model.CCATransformer import CCATransformer, Const, get_meaning_of_tokens_at_, think_about_next_token_at_, NewCCATransformer
from GPT.Model.mathexpand import add, sub, mul, div, iterate
from GPT.Model.Vector import Vector
from GPT.Data.splitToken import ChineseTokenizer

text = """
从去年起，仿佛听得有人说我是仇猫的。那根据自然是在我的那一篇《兔和猫》；这是自画招供，当然无话可说，——但倒也毫不介意。
"""

ct = ChineseTokenizer()
ct.load_model("model")

Texts = ct.tokenize(text)

print(Texts)

ccat = NewCCATransformer(Texts, 4)

height, width, tg_types = 5, 7, 9
neurons = [i*height for i in [[lambda x, parameter, nn, index: get_meaning_of_tokens_at_(ccat, add(add(x, nn.tg[index[0]][index[1]][0]), parameter), tokens)[-1]]]] * width

tokens = [Texts[1]]

rtn = RTN(
    neurons,
    weights_brush(Vector([1.0, 1.0, 1.0, 1.0]), width=width, height=height),
    [[[Vector([uniform(-1.0, 1.0) for _ in range(4)]) for _ in range(tg_types)] for _ in range(height)] for _ in range(width)],
    sr_graph_brush(width=height, height=width, tg_types=tg_types),
    [[uniform(-1.0, 1.0) for _ in range(tg_types + 1)] for _ in range(tg_types + 1)]
)

while True:
    n = rtn.forward([Vector([0.0]*4)])
    bigQ = iterate(add, n)

    next_token = think_about_next_token_at_(ccat, tokens, bigQ, 5)

    tokens.append(next_token)
    print(tokens)