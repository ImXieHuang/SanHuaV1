import sys
from pathlib import Path
from math import tanh
try:
    from .RTN import RTN, sigmoid
except:
    from RTN import RTN, sigmoid

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from convolutional import conv, pool
from Vector import Vector
from mathexpand import add, sub, mul, div

def neurons_generator(model:str, width:int = 3, height: int = 3):
    preset = {
        "any": [[lambda x, parameter, nn, index: sigmoid(x + parameter) * (x + parameter)]],
        "cnn": [[lambda x, parameter, nn, index: conv(x, parameter)],[lambda x, parameter, nn, index: pool(x)]],
        "vector": [[lambda x, parameter, nn, index: Vector([sigmoid(i) for i in x.components])]]
    }

    return [i*height for i in preset[model]] * width

def weights_brush(weight: float = 0.1, res:int = 0, width:int = 3, height: int = 3):
    if res < 0: raise(ValueError(res))
    if res == 0:
        weights = {}
        for i_l in range(width-1):
            for i_n in range(height):
                weights[(i_l, i_n)] = {}
                for j_n in range(height):
                    weights[(i_l, i_n)][(i_l+1, j_n)] = weight
    else:
        weights = {}
        for i_l in range(width-1):
            for i_n in range(height):
                weights[(i_l, i_n)] = {}
                for j_n in range(height):
                    weights[(i_l, i_n)][(i_l+1, j_n)] = weight
                if i_l % res == res-1 and i_l + res < width:
                    for res_n in range(height):
                        weights[(i_l, i_n)][(i_l+res, res_n)] = 1.0
    return [weights, [[weight]*width]*height]

def tg_brush(width:int = 3, height: int = 3, tg_types: int = 4):
    return [[[0.0 for _ in range(tg_types)] for _ in range(width)] for _ in range(height)]

def sr_graph_brush(width:int = 3, height: int = 3, tg_types: int = 4):
    return [[[0.0 for _ in range(tg_types)] for _ in range(width)] for _ in range(height)]

def tg_graph_brush(tg_types: int = 4):
    return [[0.0 for _ in range(tg_types + 1)] for _ in range(tg_types + 1)]

if __name__ == "__main__":
    from random import uniform
    rtn = RTN(neurons_generator("any"), weights_brush(1.0, 2), tg_brush(), sr_graph_brush(), tg_graph_brush())
    rtn.tg = [[[uniform(-1.0, 1.0) for _ in j] for j in i] for i in rtn.tg]
    rtn.sr_graph = [[[uniform(0.0, 1.0) for _ in j] for j in i] for i in rtn.sr_graph]
    rtn.tg_graph = [[uniform(-1.0, 1.0) for _ in i] for i in rtn.tg_graph]
    while True: print(rtn.forward([1.0,0.0,0.0]))