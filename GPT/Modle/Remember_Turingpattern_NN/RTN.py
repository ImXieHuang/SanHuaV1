import sys
from pathlib import Path
from types import LambdaType

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from mathexpand import mul, add, div, iterate

def sigmoid(x):
    e = 13580623/4996032
    return div(1, add(1, e**mul(x,-1)))

class RTN:
    __slots__ = ('neurons', 'weights', 'tg', 'sr_graph', 'tg_graph')

    def __init__(self, neurons: list[list[LambdaType]], weights: dict[tuple[int] | dict[tuple[int] | float]], tg: list[list[list[float]]], sr_graph: list[list[list[float]]] ,tg_graph: list[list[float]]):
        self.neurons = neurons
        self.weights = weights
        self.tg = [[j + [0.0] for j in i] for i in tg]
        self.sr_graph = sr_graph
        self.tg_graph = tg_graph

    def forward(self, inputs):
        nn_ans = self.nn_dynamics(inputs)

        self.tg = [[self.tg[i][j][0:-1] + [nn_ans[i][j]] for j in range(len(self.tg[i]))] for i in range(len(self.tg))]
        tg_ans = self.tg_updata()
        self.tg = [[[add(self.tg[i][j][k], tg_ans[i][j][k]) for k in range(len(self.tg[i][j]))] for j in range(len(self.tg[i]))] for i in range(len(self.tg))]

        return nn_ans[-1]

    def nn_dynamics(self, inputs) -> list[list]:
        if len(inputs) > len(self.neurons[0]): raise(ValueError(inputs))
        inputs = [inputs[i] if i < len(inputs) else 0.0 for i in range(len(self.neurons[0]))]
        answer = [[0.0 for _ in self.neurons[i]] for i in range(len(self.neurons))]

        answer[0] = [add(self.neurons[0][i]((inputs[i], self)), self.tg_dynamics(None, (0, i, len(self.tg[0][i])-1))) for i in range(len(inputs))]
        for i in range(1, len(self.neurons)):
            for j in range(len(self.neurons[i])):
                cnt = 0
                for k in [i-l for l in range(i)]:
                    for l in range(len(self.neurons[i-k])):
                        cnt = add(cnt, mul(answer[i-k][l], self.weights[(i-k, l)][(i, j)]))
                answer[i][j] = add(self.neurons[i][j]((cnt, self)), self.tg_dynamics(None, (i, j, len(self.tg[i][j])-1)))
        return answer
    
    def tg_dynamics(self, start: tuple | None, end: tuple):
        dv = 0.0
        if start is None:
            starts = []
            for i in self.weights:
                if end[:2] in self.weights[i]: starts.append(i)
            starts += [end[:2]]
            for i in starts:
                for j in range(len(self.tg[i[0]][i[1]])):
                    dv = add(dv, self.tg_dynamics((i[0], i[1], j), end))
        else:
            dv = mul(self.tg_graph[start[2]][end[2]], self.tg[start[0]][start[1]][start[2]])

        return mul(dv, self.sr_dynamics((end[0], end[1])))
    
    def sr_dynamics(self, index: tuple):
        starts = []
        ret = 0.0
        for i in self.weights:
            if index in self.weights[i]: starts.append(i)
        for i in starts:
            nsr = 0.0
            for j in range(len(self.tg[i[0]][i[1]])-1):
                nsr = add(nsr, mul(self.tg[i[0]][i[1]][j], self.sr_graph[i[0]][i[1]][j]))
            ret = add(ret, mul(nsr, self.weights[i][index]))
        return add(1, mul(-1, sigmoid(ret)))
    
    def tg_updata(self) -> list[list[list]]:
        return [[[self.tg_dynamics(None, (i, j, k)) for k in range(len(self.tg[i][j]))] for j in range(len(self.tg[i]))] for i in range(len(self.tg))]

if __name__ == "__main__":
    rtn = RTN([[lambda x: x[0], lambda x: x[0], lambda x: x[0]], [lambda x: x[0], lambda x: x[0],lambda x: x[0]]], 
              {(0,0):{(1,0): 0.5, (1,1): 1.2, (1,2): -1.0}, (0,1):{(1,0): 0.0, (1,1): 1.8, (1,2): -0.2}, (0,2):{(1,0): 1.0, (1,1): 0.5, (1,2): -1.2}},
              [[[0.0 for _ in range(4)], [0.0 for _ in range(4)], [0.0 for _ in range(4)]], [[0.0 for _ in range(4)], [0.0 for _ in range(4)], [0.0 for _ in range(4)]]],
              [[[1.0] + [0.0 for _ in range(3)], [0.0] + [0.0 for _ in range(3)], [0.0] + [0.0 for _ in range(3)]], [[0.0] + [0.0 for _ in range(3)], [0.0] + [0.0 for _ in range(3)], [0.0] + [0.0 for _ in range(3)]]],
              [[1.0, -1.0, 1.0, -1.0, 1.0], [1.0, -0.5, 1.0, -0.5, 1.0], [1.0, -1.0, 1.0, -1.0, 0.0], [1.0, -0.5, 1.0, -0.5, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]]
              )
    for i in range(100):
        ans = rtn.forward([1.0,-1.0,1])
        rtn.tg = [[[sigmoid(k) for k in j] for j in i] for i in rtn.tg]
        print(ans)