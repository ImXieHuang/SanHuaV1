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

def neurons_generator(model:str, parameter, width:int = 3, height: int = 3):
    preset = {
        "any": [[lambda x: sigmoid(x[0] + parameter) * (x[0] + parameter)]],
        "tanh": [[lambda x: tanh(x[0] + parameter) * (x[0] + parameter)]],
        "cnn": [[lambda x: conv(x[0], parameter)],[lambda x: pool(x[0])]],
        "vector": [[lambda x: Vector([sigmoid(i) for i in x[0].components])]]
    }

    return [i*height for i in preset[model]] * width

def weights_brush(res:int = 0, width:int = 3, height: int = 3):
    if res < 0: raise(ValueError(res))
    if res == 0:
        weights = {}
        for i_l in range(width-1):
            for i_n in range(height):
                weights[(i_l, i_n)] = {}
                for j_n in range(height):
                    weights[(i_l, i_n)][(i_l+1, j_n)] = 1.0
    else:
        weights = {}
        for i_l in range(width-1):
            for i_n in range(height):
                weights[(i_l, i_n)] = {}
                for j_n in range(height):
                    weights[(i_l, i_n)][(i_l+1, j_n)] = 1.0
                if i_l % res == res-1 and i_l + res < width:
                    for res_n in range(height):
                        weights[(i_l, i_n)][(i_l+res, res_n)] = 1.0
    return weights

def tg_brush(width:int = 3, height: int = 3, tg_types: int = 4):
    return [[[0.0 for _ in range(tg_types)] for _ in range(width)] for _ in range(height)]

def sr_graph_brush(width:int = 3, height: int = 3, tg_types: int = 4):
    return [[[0.0 for _ in range(tg_types)] for _ in range(width)] for _ in range(height)]

def tg_graph_brush(tg_types: int = 4):
    return [[0.0 for _ in range(tg_types + 1)] for _ in range(tg_types + 1)]

if __name__ == "__main__":
    rtn = RTN(neurons_generator("any", 0), weights_brush(), tg_brush(), sr_graph_brush(), tg_graph_brush())