import sys
from pathlib import Path
from typing import Callable

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Modle.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from Modle.mathexpand import add, sub, mul, div

class RTN_Trainer:
    def __init__(self):
        self.dx = 1e-5

    def loss_gradient(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[start][end]
    
        rtn.weights[start][end] = sub(rtn.weights[start][end], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[start][end] = original_weights

        rtn.weights[start][end] = add(rtn.weights[start][end], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[start][end] = original_weights

        d_loss = div(sub(add_loss, sub_loss), 2)

        return div(d_loss, self.dx)
    
    def traverse_weight_for_(self, function: callable, rtn: RTN):
        ret = {}
        for i,j in zip(list(rtn.weights.keys()), list(rtn.weights.values())):
            ret[i] = {}
            for k,l in zip(list(j.keys()), list(j.values())):
                ret[i][k] = function(i, k, l)
        return ret
    
    def static_trainer(self, inputs: list[list], outputs: list[list], lossfunction: callable, lambdafunction: callable, r: float, rtn: RTN):
        def gradfunction(start, end, weight):
            return self.loss_gradient(start, end, ip, lambda x: lossfunction(x, op) + L, rtn)
        def iteration(start, end, weight):
            rtn.weights[start][end] = sub(rtn.weights[start][end], mul(r, g[start][end]))

        L = 0.0
        for i in list(rtn.weights.values()):
            for j in list(i.values()):
                L = add(L, lambdafunction(j))
    
        for ip,op in zip(inputs, outputs):
            g = t.traverse_weight_for_(gradfunction, rtn)
            t.traverse_weight_for_(iteration, rtn)

if __name__ == "__main__":
    t = RTN_Trainer()
    rtn = RTN(neurons_generator('any', 0),
              weights_brush(), 
              tg_brush(), 
              sr_graph_brush(), 
              tg_graph_brush()
              )
    def loss(x, op):
        cnt = 0.0
        for i,j in zip(x,op):
            cnt += sub(i, j) ** 2
        return cnt
    def lambda_(weight):
        return weight ** 2

    for i in range(10):
        t.static_trainer([[i,0,0] for i in range(4)], [[0,0,i] for i in range(4)], loss, lambda_, 0.1, rtn)

    print("Training is end")
    
    for i in range(10):
        print(rtn.nn_dynamics([i, 0, 0])[-1])