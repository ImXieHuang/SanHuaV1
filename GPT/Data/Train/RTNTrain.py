import sys
from pathlib import Path
from typing import Callable
from random import uniform

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Modle.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from Modle.mathexpand import add, sub, mul, div

class RTN_Trainer:
    def __init__(self):
        self.dx = 1e-5

    def weight_loss_gradient(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[0][start][end]
    
        rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        rtn.weights[0][start][end] = add(rtn.weights[0][start][end], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        d_loss = div(sub(add_loss, sub_loss), 2)

        return div(d_loss, self.dx)
    
    def parameter_loss_gradient(self, index: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[1][index[0]][index[1]]
    
        rtn.weights[1][index[0]][index[1]] = sub(rtn.weights[1][index[0]][index[1]], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[1][index[0]][index[1]] = original_weights

        rtn.weights[1][index[0]][index[1]] = add(rtn.weights[1][index[0]][index[1]], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[1][index[0]][index[1]] = original_weights

        d_loss = div(sub(add_loss, sub_loss), 2)

        return div(d_loss, self.dx)
    
    def traverse_weight_for_(self, function: callable, rtn: RTN):
        ret = {}
        for i,j in zip(list(rtn.weights[0].keys()), list(rtn.weights[0].values())):
            ret[i] = {}
            for k,l in zip(list(j.keys()), list(j.values())):
                ret[i][k] = function(i, k, l)
        return ret
    
    def static_trainer(self, inputs: list[list], outputs: list[list], lossfunction: callable, lambdafunction: callable, r: float, dorpout: float, rtn: RTN):
        L = 0.0
        for i in list(rtn.weights[0].values()):
            for j in list(i.values()):
                L = add(L, lambdafunction(j))
        for i in rtn.weights[1]:
            for j in i:
                L = add(L, lambdafunction(j))
        
        def Wgradfunction(start, end, weight):
            return self.weight_loss_gradient(start, end, ip, lambda x: lossfunction(x, op) + L, rtn)
        def Pgradfunction(index, _, weight):
            return self.parameter_loss_gradient(index, ip, lambda x: lossfunction(x, op) + L, rtn)
        def Witeration(start, end, weight):
            if uniform(0.0,1.0) < dorpout: return
            rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], mul(r, wg[start][end]))
        def Piteration(index, _, weight):
            if uniform(0.0,1.0) < dorpout: return
            rtn.weights[1][index[0]][index[1]] = sub(rtn.weights[1][index[0]][index[1]], mul(r, pg[index][_]))
    
        for ip,op in zip(inputs, outputs):
            wg = t.traverse_weight_for_(Wgradfunction, rtn)
            pg = t.traverse_weight_for_(Pgradfunction, rtn)
            t.traverse_weight_for_(Witeration, rtn)
            t.traverse_weight_for_(Piteration, rtn)

if __name__ == "__main__":
    t = RTN_Trainer()
    rtn = RTN(neurons_generator('any'),
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
        return 0.1 * weight ** 2

    for i in range(10):
        t.static_trainer([[i,0,0] for i in range(4)], [[0,0,i] for i in range(4)], loss, lambda_, 0.05, 0.2, rtn)
        print(rtn.nn_dynamics([1.0, 0, 0])[-1])

    print("Training is end")
    
    for i in range(10):
        print(rtn.nn_dynamics([i, 0, 0])[-1])