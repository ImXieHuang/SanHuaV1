import sys
from pathlib import Path
from typing import Callable
from random import uniform

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Model.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from Model.mathexpand import add, sub, mul, div, iterate

class RTN_Trainer:
    def __init__(self, dx = None, maxpid_i = None):
        self.dx = dx or 1e-5
        self.pid_i = 0.0
        self.pid_d = 0.0
        self.maxpid_i = maxpid_i or 10

    def pid(self, kp, ki, kd, loss):
        ret = kp * loss + ki * self.pid_i + kd * (loss - self.pid_d)
        self.pid_i = max(min(self.pid_i + loss, self.maxpid_i), -self.maxpid_i)
        self.pid_d = loss

        return ret 

    def weight_loss_gradient(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[0][start][end]
    
        rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        rtn.weights[0][start][end] = add(rtn.weights[0][start][end], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        return div(div(sub(add_loss, sub_loss), 2), self.dx)
    
    def parameter_loss_gradient(self, index: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[1][index[0]][index[1]]
    
        rtn.weights[1][index[0]][index[1]] = sub(rtn.weights[1][index[0]][index[1]], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[1][index[0]][index[1]] = original_weights

        rtn.weights[1][index[0]][index[1]] = add(rtn.weights[1][index[0]][index[1]], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[1][index[0]][index[1]] = original_weights

        return div(div(sub(add_loss, sub_loss), 2), self.dx)
    
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
            Lam = 0.0
            Lambdacnt = 0
            for i in list(rtn.weights[0].values()):
                for j in list(i.values()):
                    Lam = add(Lam, lambdafunction(j))
                    Lambdacnt += 1
            for i in rtn.weights[1]:
                for j in i:
                    Lam = add(Lam, lambdafunction(j))
                    Lambdacnt += 1

            Lam = div(Lam, Lambdacnt)
            
            def Wgradfunction(start, end, weight):
                return self.weight_loss_gradient(start, end, ip, lambda x: lossfunction(x, op) + Lam, rtn)
            def Pgradfunction(index, _, weight):
                return self.parameter_loss_gradient(index, ip, lambda x: lossfunction(x, op) + Lam, rtn)
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

    def rtn_sample(self, rtn:RTN, inputs, maxt: int = 30, sample_rate: int = 5):
        rtn_sampling = []
        sample_step = int(maxt / sample_rate)
        
        for i in range(maxt):
            ans = rtn.forward(inputs)
            if i % sample_step == 0:
                rtn_sampling.append(ans)
            
        return rtn_sampling

    def fitting_pid_for_(self, rtn_sampling, maxt: int = 30, sample_rate: int = 5, r: float = 0.03):
        if not rtn_sampling:
            return [[0.0, 0.0, 0.0]]
            
        rtn_sampling = [list(row) for row in zip(*rtn_sampling)]

        kp = [0.01 for _ in rtn_sampling]
        ki = [0.001 for _ in rtn_sampling]
        kd = [0.005 for _ in rtn_sampling]
        
        KP_MIN, KP_MAX = -1.0, 1.0
        KI_MIN, KI_MAX = -0.1, 0.1
        KD_MIN, KD_MAX = -0.1, 0.1
        
        sample_indices = list(range(0, maxt, int(maxt/sample_rate)))
        
        for _ in range(200):
            total_loss = 0.0
            for pipe in range(len(rtn_sampling)):
                target = rtn_sampling[pipe][-1]
                
                for param_idx in range(3):
                    self.pid_i = 0.0
                    self.pid_d = 0.0
                    pid_sampling = []
                    for i in range(maxt):
                        error = target - (pid_sampling[-1] if pid_sampling else 0)
                        kp_tmp = kp[pipe] + (self.dx if param_idx==0 else 0)
                        ki_tmp = ki[pipe] + (self.dx if param_idx==1 else 0)
                        kd_tmp = kd[pipe] + (self.dx if param_idx==2 else 0)
                        ans = self.pid(kp_tmp, ki_tmp, kd_tmp, error)
                        if i in sample_indices:
                            pid_sampling.append(ans)
                    
                    sub_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], pid_sampling)])
                    
                    self.pid_i = 0.0
                    self.pid_d = 0.0
                    pid_sampling = []
                    for i in range(maxt):
                        error = target - (pid_sampling[-1] if pid_sampling else 0)
                        kp_tmp = kp[pipe] - (self.dx if param_idx==0 else 0)
                        ki_tmp = ki[pipe] - (self.dx if param_idx==1 else 0)
                        kd_tmp = kd[pipe] - (self.dx if param_idx==2 else 0)
                        ans = self.pid(kp_tmp, ki_tmp, kd_tmp, error)
                        if i in sample_indices:
                            pid_sampling.append(ans)
                    
                    add_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], pid_sampling)])
                    
                    gradient = (add_loss - sub_loss) / (2 * self.dx)
                    gradient = max(min(gradient, 1.0), -1.0)
                    
                    if param_idx == 0:
                        kp[pipe] -= gradient * r
                        kp[pipe] = max(min(kp[pipe], KP_MAX), KP_MIN)
                    elif param_idx == 1:
                        ki[pipe] -= gradient * r
                        ki[pipe] = max(min(ki[pipe], KI_MAX), KI_MIN)
                    else:
                        kd[pipe] -= gradient * r
                        kd[pipe] = max(min(kd[pipe], KD_MAX), KD_MIN)
                
                self.pid_i = 0.0
                self.pid_d = 0.0
                final_sampling = []
                for t in range(maxt):
                    error = target - (final_sampling[-1] if final_sampling else 0)
                    control = self.pid(kp[pipe], ki[pipe], kd[pipe], error)
                    if t in sample_indices:
                        final_sampling.append(control)
                
                current_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], final_sampling)])
                total_loss += current_loss
        return [list(row) for row in zip(kp, ki, kd)]

    def rtn_sample(self, rtn:RTN, inputs, maxt: int = 30, sample_rate: int = 5):
        rtn_sampling = []
        sample_step = int(maxt / sample_rate)
        
        for i in range(maxt):
            ans = rtn.forward(inputs)
            if i % sample_step == 0:
                rtn_sampling.append(ans)
            
        return rtn_sampling

    def fitting_pid_for_(self, rtn_sampling, maxt: int = 30, sample_rate: int = 5, r: float = 0.03):
        if not rtn_sampling:
            return [[0.0, 0.0, 0.0]]
            
        rtn_sampling = [list(row) for row in zip(*rtn_sampling)]

        kp = [0.01 for _ in rtn_sampling]
        ki = [0.001 for _ in rtn_sampling]
        kd = [0.005 for _ in rtn_sampling]
        
        KP_MIN, KP_MAX = -1.0, 1.0
        KI_MIN, KI_MAX = -0.1, 0.1
        KD_MIN, KD_MAX = -0.1, 0.1
        
        sample_indices = list(range(0, maxt, int(maxt/sample_rate)))
        
        for _ in range(200):
            total_loss = 0.0
            for pipe in range(len(rtn_sampling)):
                target = rtn_sampling[pipe][-1]
                
                for param_idx in range(3):
                    self.pid_i = 0.0
                    self.pid_d = 0.0
                    pid_sampling = []
                    for i in range(maxt):
                        error = target - (pid_sampling[-1] if pid_sampling else 0)
                        kp_tmp = kp[pipe] + (self.dx if param_idx==0 else 0)
                        ki_tmp = ki[pipe] + (self.dx if param_idx==1 else 0)
                        kd_tmp = kd[pipe] + (self.dx if param_idx==2 else 0)
                        ans = self.pid(kp_tmp, ki_tmp, kd_tmp, error)
                        if i in sample_indices:
                            pid_sampling.append(ans)
                    
                    sub_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], pid_sampling)])
                    
                    self.pid_i = 0.0
                    self.pid_d = 0.0
                    pid_sampling = []
                    for i in range(maxt):
                        error = target - (pid_sampling[-1] if pid_sampling else 0)
                        kp_tmp = kp[pipe] - (self.dx if param_idx==0 else 0)
                        ki_tmp = ki[pipe] - (self.dx if param_idx==1 else 0)
                        kd_tmp = kd[pipe] - (self.dx if param_idx==2 else 0)
                        ans = self.pid(kp_tmp, ki_tmp, kd_tmp, error)
                        if i in sample_indices:
                            pid_sampling.append(ans)
                    
                    add_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], pid_sampling)])
                    
                    gradient = (add_loss - sub_loss) / (2 * self.dx)
                    gradient = max(min(gradient, 1.0), -1.0)
                    
                    if param_idx == 0:
                        kp[pipe] -= gradient * r
                        kp[pipe] = max(min(kp[pipe], KP_MAX), KP_MIN)
                    elif param_idx == 1:
                        ki[pipe] -= gradient * r
                        ki[pipe] = max(min(ki[pipe], KI_MAX), KI_MIN)
                    else:
                        kd[pipe] -= gradient * r
                        kd[pipe] = max(min(kd[pipe], KD_MAX), KD_MIN)
                
                self.pid_i = 0.0
                self.pid_d = 0.0
                final_sampling = []
                for t in range(maxt):
                    error = target - (final_sampling[-1] if final_sampling else 0)
                    control = self.pid(kp[pipe], ki[pipe], kd[pipe], error)
                    if t in sample_indices:
                        final_sampling.append(control)
                
                current_loss = sum([(rtn_val - pid_val)**2 for rtn_val, pid_val in zip(rtn_sampling[pipe], final_sampling)])
                total_loss += current_loss
        return [list(row) for row in zip(kp, ki, kd)]

    def sr_graph_loss_gradient(self, index: tuple, input, target_output, target_pid: list, samplingloss: callable, staticloss: callable, rtn: RTN, maxt: int = 30, sample_rate: int = 5):     
        print(f"  compute sr gradient {index = }")
        original_sr_graph = rtn.sr_graph[index[0]][index[1]][index[2]]
        
        rtn.sr_graph[index[0]][index[1]][index[2]] = add(rtn.sr_graph[index[0]][index[1]][index[2]], self.dx)
        sampling = self.rtn_sample(rtn, input, maxt, sample_rate)
        pid_params = self.fitting_pid_for_(sampling, maxt, sample_rate)
        rtn.sr_graph[index[0]][index[1]][index[2]] = original_sr_graph
        add_loss = add(samplingloss(pid_params, target_pid), staticloss(sampling[-1], target_output))

        rtn.sr_graph[index[0]][index[1]][index[2]] = sub(rtn.sr_graph[index[0]][index[1]][index[2]], self.dx)
        sampling = self.rtn_sample(rtn, input, maxt, sample_rate)
        pid_params = self.fitting_pid_for_(sampling, maxt, sample_rate)
        rtn.sr_graph[index[0]][index[1]][index[2]] = original_sr_graph
        sub_loss = add(samplingloss(pid_params, target_pid), staticloss(sampling[-1], target_output))

        return div(div(sub(add_loss, sub_loss), 2), self.dx)

    def tg_graph_loss_gradient(self, i: int, j: int, input, target_output, target_pid: list, samplingloss: callable, staticloss: callable, rtn: RTN, maxt: int = 30, sample_rate: int = 5):
        print(f"  compute tg gradient tg.{i = } tg.{j = }")
        original_tg_graph = rtn.tg_graph[i][j]
        
        rtn.tg_graph[i][j] = add(rtn.tg_graph[i][j], self.dx)
        sampling = self.rtn_sample(rtn, input, maxt, sample_rate)
        pid_params = self.fitting_pid_for_(sampling, maxt, sample_rate)
        rtn.tg_graph[i][j] = original_tg_graph
        add_loss = add(samplingloss(pid_params, target_pid), staticloss(sampling[-1], target_output))

        rtn.tg_graph[i][j] = sub(rtn.tg_graph[i][j], self.dx)
        sampling = self.rtn_sample(rtn, input, maxt, sample_rate)
        pid_params = self.fitting_pid_for_(sampling, maxt, sample_rate)
        rtn.tg_graph[i][j] = original_tg_graph
        sub_loss = add(samplingloss(pid_params, target_pid), staticloss(sampling[-1], target_output))

        return div(div(sub(add_loss, sub_loss), 2), self.dx)

    def rtn_sampling_trainer(self, inputs: list[list], outputs: list[list], target_pids: list[list], samplinglossfunction: callable, staticlossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dorpout: float, rtn: RTN, maxt: int = 30, sample_rate: int = 5):
        original_tg = rtn.tg
        original_sr_graph = rtn.sr_graph
        original_tg_graph = rtn.tg_graph
        
        try:
            Lam = 0.0
            Lambdacnt = 0
            
            for i in rtn.tg_graph:
                for j in i:
                    Lam = add(Lam, lambdafunction(j))
                    Lambdacnt += 1
            for i in rtn.sr_graph:
                for j in i:
                    for k in j:
                        Lam = add(Lam, lambdafunction(k))
                        Lambdacnt += 1

            Lam = div(Lam, Lambdacnt) if Lambdacnt > 0 else 0.0
            
            for ip, op, target_pid in zip(inputs, outputs, target_pids):
                print(f"Training {ip} -> {op} with target PID {target_pid}")

                for i in range(len(rtn.sr_graph)):
                    for j in range(len(rtn.sr_graph[i])):
                        for k in range(len(rtn.sr_graph[i][j])):
                            if uniform(0.0,1.0) >= dorpout:
                                gradient = self.sr_graph_loss_gradient((i, j, k), ip, op, target_pid, samplinglossfunction, lambda x, y: add(staticlossfunction(x, y), Lam), rtn, maxt, sample_rate)
                                rtn.sr_graph[i][j][k] = sub(rtn.sr_graph[i][j][k], min(max(-maxdw, mul(r, gradient)), maxdw))
                
                for i in range(len(rtn.tg_graph)):
                    for j in range(len(rtn.tg_graph[i])):
                        if uniform(0.0,1.0) >= dorpout:
                            gradient = self.tg_graph_loss_gradient(i, j, ip, op, target_pid, samplinglossfunction, lambda x, y: add(staticlossfunction(x, y), Lam), rtn, maxt, sample_rate)
                            rtn.tg_graph[i][j] = sub(rtn.tg_graph[i][j], min(max(-maxdw, mul(r, gradient)), maxdw))

        except Exception as error:
            rtn.tg = original_tg
            rtn.sr_graph = original_sr_graph
            rtn.tg_graph = original_tg_graph
            return error

        return True

if __name__ == "__main__":
    ipt = input(
"""Please choose a test content:
1) static trainer
2) PID controller
3) rtn_sampling trainer
4) None
>> """
)
    print("\n"+"="*20+"\n")
    if  ipt == "1":
        print("# static trainer start\n")

        t = RTN_Trainer()
        rtn = RTN(neurons_generator('any'),
                weights_brush(0.0, 0.1, 2), 
                tg_brush(), 
                sr_graph_brush(), 
                tg_graph_brush()
                )
        
        ip = [[2*i+1,0,0] for i in range(4)]+[[0,2*i+1,0] for i in range(4)]+[[0,0,2*i+1] for i in range(4)]
        op = [[0,0,2*i+1] for i in range(4)]+[[0,0,0] for _ in range(8)]

        print("data:")
        
        for i,j in zip(ip, op):
            print(f"{i} -> {j}")

        def static_loss(x, op):
            loss = 0.0
            for o, t in zip(x, op):
                diff = sub(o, t)
                loss = add(loss, mul(diff, diff))
            return loss
        def lambda_loss(weight):
            return mul(mul(r, 0.01), mul(weight, weight))

        r = 0.03

        print(f"\n{r = }\n")

        for cnt in range(10):
            print(f"## turn {cnt+1}")

            print(t.static_trainer(ip, op, static_loss, lambda_loss, r, 0.05, 0.2, rtn))

            error = 0.0
            for i, o in zip(ip, op):
                x = rtn.nn_dynamics(i)
                error = add(error, static_loss(x, o))

            print(f"{error = }, {r = }\n{rtn.nn_dynamics([1, 0, 0])[-1] = }\n")

        print("Training is end")
        
        print("\ntest data:")
        for i in range(10):
            print(f"[{i},0,0] -> {rtn.nn_dynamics([i, 0, 0])[-1]}")

    elif ipt == "2":
        g = 9.8
        m = 10

        print("# PID controller\n")
        print("| d^2/dt^2 ym = -gm + alpha <- Process Value")
        print("| setpoint y = 10")
        print(f"| {g = } {m = }")

        def loss(x, y): return x-y

        print("\nrun Simulation:")
        
        pid = RTN_Trainer().pid

        y = 0
        v = 0

        for i in range(200):
            alpha = pid(20.0, 9.8, 5.0, loss(10, y))
            v += -g + alpha/m
            y += v

            print(y)

    elif ipt == "3":
        print("# sampling trainer start\n")

        t = RTN_Trainer()
        rtn = RTN(neurons_generator('any'),
                weights_brush(0.0, 0.1, 2), 
                tg_brush(), 
                sr_graph_brush(), 
                tg_graph_brush()
                )
        
        ip = [[2*i+1,0,0] for i in range(4)]+[[0,2*i+1,0] for i in range(4)]+[[0,0,2*i+1] for i in range(4)]
        op = [[0,0,2*i+1] for i in range(4)]+[[0,0,0] for _ in range(8)]
        tp = [[0.1, 0.01, 0.05] for _ in range(len(ip))]
        
        print("data:")
        for i,j,p in zip(ip, op, tp):
            print(f"{i} -> {j}  target PID: {p}")
        
        def sample_loss(pid_params, target_pid):
            loss = 0.0
            if isinstance(pid_params, list) and len(pid_params) > 0:
                if isinstance(pid_params[0], list):
                    for p, t in zip(pid_params[0], target_pid):
                        diff = sub(p, t)
                        loss = add(loss, mul(diff, diff))
                else:
                    for p, t in zip(pid_params, target_pid):
                        diff = sub(p, t)
                        loss = add(loss, mul(diff, diff))
            return loss
        
        def static_loss(x, op):
            loss = 0.0
            for o, t in zip(x, op):
                diff = sub(o, t)
                loss = add(loss, mul(diff, diff))
            return loss
        
        def lambda_loss(weight):
            return mul(mul(r, 0.01), mul(weight, weight))
        
        r = 0.03
        maxdw = 0.05
        dropout = 0.2
        maxt = 30
        sample_rate = 5
        
        print(f"\n{r = }")
        print(f"{maxdw = }")
        print(f"{dropout = }")
        print(f"{maxt = }")
        print(f"{sample_rate = }\n")
        
        print("pre-training test:")
        for i, o in zip(ip[:3], op[:3]):
            sampling = t.rtn_sample(rtn, i, maxt, sample_rate)
            output_val = sampling[-1][0] if sampling and sampling[-1] else 0
            target_val = o[0] if o else 0
            print(f"input {i} -> output {output_val:.6f}, target {target_val}")
        
        print("\nstarting training...\n")
        
        for epoch in range(5):
            print(f"## turn {epoch+1}")
            
            result = t.rtn_sampling_trainer(
                ip, op, tp,
                sample_loss,
                static_loss,
                lambda_loss,
                r, maxdw, dropout,
                rtn,
                maxt, sample_rate
            )
            
            if isinstance(result, Exception):
                print(f"training error: {result}")
                break
            
            total_error = 0.0
            for i, o in zip(ip, op):
                sampling = t.rtn_sample(rtn, i, maxt, sample_rate)
                if sampling and sampling[-1]:
                    error = static_loss(sampling[-1], o)
                    total_error = add(total_error, error)
            
            test_input = [1, 0, 0]
            test_output = rtn.nn_dynamics(test_input)[-1]
            test_val = test_output[0] if test_output else 0
            
            print(f"{total_error = }, {r = }")
            print(f"{rtn.nn_dynamics([1, 0, 0])[-1] = }\n")
        
        print("training completed!\n")
        
        print("test data:")
        for i in range(5):
            test_input = [i, 0, 0]
            test_output = rtn.nn_dynamics(test_input)[-1]
            test_val = test_output[0] if test_output else 0
            print(f"[{i},0,0] -> {test_val}")
        
        print("\nfitted PID parameters for each input:")
        for i in ip[:3]:
            sampling = t.rtn_sample(rtn, i, maxt, sample_rate)
            pid_params = t.fitting_pid_for_(sampling)
            print(f"input {i} -> PID: {pid_params}")

    else: print(None)
