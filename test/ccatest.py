import sys
from pathlib import Path
from random import uniform

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from GPT.Modle.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from GPT.Modle.CCATransformer import CCATransformer, Const, get_meaning_of_tokens_at_, think_about_next_token_at_, NewCCATransformer
from GPT.Modle.mathexpand import add, sub, mul, div, iterate
from GPT.Modle.Vector import Vector
from GPT.Data.splitToken import ChineseTokenizer

text = """
从去年起，仿佛听得有人说我是仇猫的。那根据自然是在我的那一篇《兔和猫》；这是自画招供，当然无话可说，——但倒也毫不介意。
"""

ct = ChineseTokenizer()
ct.load_model("model")

Texts = ct.tokenize(text)

print(Texts)

cat = NewCCATransformer(Texts)

height, width, tg_types = 3, 3, 4
neurons = [i*height for i in [[lambda x, parameter, nn, index: get_meaning_of_tokens_at_(cat, add(x, Vector(nn.tg[index[0]][index[1]][:4])), tokens)[-1]]]] * width

tokens = [Texts[1]]

rtn = RTN(
    neurons,
    weights_brush(Vector([1.0, 1.0, 1.0, 1.0])),
    [[[uniform(-1.0, 1.0) for _ in range(tg_types)] for _ in range(width)] for _ in range(height)],
    sr_graph_brush(),
    [[uniform(-1.0, 1.0) for _ in range(tg_types + 1)] for _ in range(tg_types + 1)]
)

while True:
    n = rtn.nn_dynamics([Vector([0.0]*4)]*3)[-1]
    bigQ = iterate(add, n)

    next_token = think_about_next_token_at_(cat, tokens, bigQ, 5)

    tokens.append(next_token)
    print(tokens)