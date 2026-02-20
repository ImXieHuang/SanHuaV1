import sys
from pathlib import Path

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Modle.Remember_Turingpattern_NN import RTN, operations
from Modle.mathexpand import add, sub, mul, div

class RTN_Trainer:
    def __init__(self):
        self.dx = 1e-5

    def gradient_for_(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[start][end]
    
        rtn.weights[start][end] = sub(rtn.weights[start][end], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[start][end] = original_weights

        rtn.weights[start][end] = add(rtn.weights[start][end], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[start][end] = original_weights

        d_loss = div(sub(add_loss, sub_loss), 2)

        return div(d_loss, self.dx)
    
if __name__ == "__main__":
    t = RTN_Trainer()