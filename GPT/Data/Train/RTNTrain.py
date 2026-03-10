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
    
    def static_trainer(self, inputs: list[list], outputs: list[list], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dorpout: float, rtn: RTN):
        original_tg = rtn.tg
        original_weights = rtn.weights
        try:
            rtn.tg = [[[0.0 for _ in j] for j in i] for i in rtn.tg]
            L = 0.0
            Lambdacnt = 0
            for i in list(rtn.weights[0].values()):
                for j in list(i.values()):
                    L = add(L, lambdafunction(j))
                    Lambdacnt += 1
            for i in rtn.weights[1]:
                for j in i:
                    L = add(L, lambdafunction(j))
                    Lambdacnt += 1

            L = div(L, Lambdacnt)
            
            def Wgradfunction(start, end, weight):
                return self.weight_loss_gradient(start, end, ip, lambda x: lossfunction(x, op) + L, rtn)
            def Pgradfunction(index, _, weight):
                return self.parameter_loss_gradient(index, ip, lambda x: lossfunction(x, op) + L, rtn)
            def Witeration(start, end, weight):
                if uniform(0.0,1.0) < dorpout: return
                rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], min(max(-maxdw, mul(r, wg[start][end])), maxdw))
            def Piteration(index, _, weight):
                if uniform(0.0,1.0) < dorpout: return
                rtn.weights[1][index[0]][index[1]] = sub(rtn.weights[1][index[0]][index[1]], min(max(-maxdw, mul(r, pg[index][_])), maxdw))
        
            for ip,op in zip(inputs, outputs):
                print(f"Training {ip} -> {op}")
                wg = self.traverse_weight_for_(Wgradfunction, rtn)
                pg = self.traverse_weight_for_(Pgradfunction, rtn)
                self.traverse_weight_for_(Witeration, rtn)
                self.traverse_weight_for_(Piteration, rtn)
        except Exception as error:
            rtn.weights = original_weights
            rtn.tg = original_tg
            return error

        rtn.tg = original_tg
        return True

if __name__ == "__main__":
    t = RTN_Trainer()
    rtn = RTN(neurons_generator('any'),
              weights_brush(0.0, 0.1, 2), 
              tg_brush(), 
              sr_graph_brush(), 
              tg_graph_brush()
              )
    
    ip = [[2*i+1,0,0] for i in range(4)]+[[0,2*i+1,0] for i in range(4)]+[[0,0,2*i+1] for i in range(4)]
    op = [[0,0,2*i+1] for i in range(4)]+[[0,0,0] for _ in range(8)]

    def loss(x, op):
        loss = 0.0
        for o, t in zip(x, op):
            diff = sub(o, t)
            loss = add(loss, mul(diff, diff))
        return loss
    def lambda_(weight):
        return mul(mul(r, 0.01), mul(weight, weight))
    
    r = 0.1

    for i in range(100):
        print(t.static_trainer(ip, op, loss, lambda_, r, 1.0, 0.2, rtn))

        error = 0.0
        for i, o in zip(ip, op):
            x = rtn.nn_dynamics(i)
            error = add(error, loss(x, o))
        error = sum(error)

        print(f"{error = }, {r = }\n{rtn.nn_dynamics([1, 0, 0])[-1] = }\n")

    print("Training is end")
    
    for i in range(10):
        print(rtn.nn_dynamics([i, 0, 0])[-1])