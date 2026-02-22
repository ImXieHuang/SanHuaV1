import sys
from pathlib import Path

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Modle.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from Modle.mathexpand import add, sub, mul, div

class RTN_Trainer:
    def __init__(self):
        self.dx = 1e-5

    def gradient(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
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
    
    def static_trainer(self, inputs: list[list], outputs: list[list], rtn: RTN, r = 0.1, target = 1.0):
        def iteration(i,j,k):
            rtn.weights[i][j] = sub(rtn.weights[i][j], mul(div(g[i][j], len(inputs)), r))
        def loss(x, y):
            return sum([(i-j)**2 for i,j in zip(y, x)])
        def regularization():
            cnt = 0.0
            for i in list(rtn.weights.values()):
                for j in list(i.values()):
                    cnt += j**2
            return cnt
        
        generation = 0
        
        while True:
            generation += 1

            for i,j in zip(inputs, outputs):
                g = t.traverse_weight_for_(lambda a,b,c: t.gradient(a, b, i, lambda x: loss(j, x)+regularization(), rtn), rtn)
                t.traverse_weight_for_(iteration, rtn)
            
            error = 0.0

            for i,j in zip(inputs, outputs):
                ans = rtn.nn_dynamics(i)[-1]
                error = add(error, loss(j, ans))

            print(f"{generation} generation: {error = }")
            if error <= target:
                break
        
        return error

if __name__ == "__main__":
    t = RTN_Trainer()
    rtn = RTN(neurons_generator('any', 0), weights_brush(), tg_brush(), sr_graph_brush(), tg_graph_brush())

    t.static_trainer([[i,0,0] for i in range(10)], [[0,0,1] for i in range(10)], rtn, 0.1, 1.0)

    print("Training is end")
    
    for i in range(10):
        print(rtn.forward([i, 0, 0]))