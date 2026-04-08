import sys
from pathlib import Path
from random import uniform
import json
import pickle
from datetime import datetime
from typing import Any, List

udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)

from Model.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from Model.mathexpand import add, sub, mul, div

class RTN_Trainer:
    def __init__(self, dx: Any = None, maxpid_i: Any = None):
        self.dx = dx or 1e-5
        self.pid_i = 0.0
        self.pid_d = 0.0
        self.maxpid_i = maxpid_i or 10
        self.models_dir = self.get_models_dir()
    
    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def pid(self, kp, ki, kd, loss):
        ret = kp * loss + ki * self.pid_i + kd * (loss - self.pid_d)
        self.pid_i = max(min(self.pid_i + loss, self.maxpid_i), -self.maxpid_i)
        self.pid_d = loss
        return ret 

    def weight_loss_derivative(self, start: tuple, end: tuple, inputs: list, loss: callable, rtn: RTN):
        original_weights = rtn.weights[0][start][end]
    
        rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], self.dx)
        sub_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        rtn.weights[0][start][end] = add(rtn.weights[0][start][end], self.dx)
        add_loss = loss(rtn.nn_dynamics(inputs)[-1])
        rtn.weights[0][start][end] = original_weights

        return div(div(sub(add_loss, sub_loss), 2), self.dx)
    
    def parameter_loss_derivative(self, index: tuple, inputs: list, loss: callable, rtn: RTN):
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
    
    def static_trainer(self, inputs: List[List], outputs: List[List], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, rtn: RTN):
        original_tg = rtn.tg
        original_weights = rtn.weights
        try:
            rtn.tg = [[[0.0 for _ in j] for j in i] for i in rtn.tg]
            
            for ip, op in zip(inputs, outputs):
                print(f"Training {ip} -> {op}")
                
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
                Lam = div(Lam, Lambdacnt) if Lambdacnt > 0 else 0.0
                
                wg = {}
                for start in list(rtn.weights[0].keys()):
                    wg[start] = {}
                    for end in list(rtn.weights[0][start].keys()):
                        def loss_with_reg(x):
                            return add(lossfunction(x, op), Lam)
                        wg[start][end] = self.weight_loss_derivative(start, end, ip, loss_with_reg, rtn)
                
                pg = {}
                for idx in range(len(rtn.weights[1])):
                    pg[idx] = {}
                    for jdx in range(len(rtn.weights[1][idx])):
                        def loss_with_reg(x):
                            return add(lossfunction(x, op), Lam)
                        pg[idx][jdx] = self.parameter_loss_derivative((idx, jdx), ip, loss_with_reg, rtn)
                
                for start in list(wg.keys()):
                    for end in list(wg[start].keys()):
                        if uniform(0.0, 1.0) >= dropout:
                            delta = mul(r, wg[start][end])
                            if delta > maxdw:
                                delta = maxdw
                            if delta < -maxdw:
                                delta = -maxdw
                            rtn.weights[0][start][end] = sub(rtn.weights[0][start][end], delta)
                
                for idx in list(pg.keys()):
                    for jdx in list(pg[idx].keys()):
                        if uniform(0.0, 1.0) >= dropout:
                            delta = mul(r, pg[idx][jdx])
                            if delta > maxdw:
                                delta = maxdw
                            if delta < -maxdw:
                                delta = -maxdw
                            rtn.weights[1][idx][jdx] = sub(rtn.weights[1][idx][jdx], delta)

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
                    
                    derivative = (add_loss - sub_loss) / (2 * self.dx)
                    derivative = max(min(derivative, 1.0), -1.0)
                    
                    if param_idx == 0:
                        kp[pipe] -= derivative * r
                        kp[pipe] = max(min(kp[pipe], KP_MAX), KP_MIN)
                    elif param_idx == 1:
                        ki[pipe] -= derivative * r
                        ki[pipe] = max(min(ki[pipe], KI_MAX), KI_MIN)
                    else:
                        kd[pipe] -= derivative * r
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

    def sr_graph_loss_derivative(self, index: tuple, input, target_output, target_pid: list, samplingloss: callable, staticloss: callable, rtn: RTN, maxt: int = 30, sample_rate: int = 5):     
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

    def tg_graph_loss_derivative(self, i: int, j: int, input, target_output, target_pid: list, samplingloss: callable, staticloss: callable, rtn: RTN, maxt: int = 30, sample_rate: int = 5):
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

    def rtn_sampling_trainer(self, inputs: List[List], outputs: List[List], target_pids: List[List], samplinglossfunction: callable, staticlossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, rtn: RTN, maxt: int = 30, sample_rate: int = 5):
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
            
            if len(inputs) != 1 or len(outputs) != 1 or len(target_pids) != 1:
                raise ValueError("This trainer only accepts one pair of input-output data")
            
            ip = inputs[0]
            op = outputs[0]
            target_pid = target_pids[0]

            for i in range(len(rtn.sr_graph)):
                for j in range(len(rtn.sr_graph[i])):
                    for k in range(len(rtn.sr_graph[i][j])):
                        if uniform(0.0,1.0) >= dropout:
                            def loss_with_reg_sampling(pid_params, target):
                                return add(samplinglossfunction(pid_params, target), Lam)
                            def loss_with_reg_static(x, y):
                                return add(staticlossfunction(x, y), Lam)
                            derivative = self.sr_graph_loss_derivative((i, j, k), ip, op, target_pid, loss_with_reg_sampling, loss_with_reg_static, rtn, maxt, sample_rate)
                            delta = mul(r, derivative)
                            if delta > maxdw:
                                delta = maxdw
                            if delta < -maxdw:
                                delta = -maxdw
                            rtn.sr_graph[i][j][k] = sub(rtn.sr_graph[i][j][k], delta)
            
            for i in range(len(rtn.tg_graph)):
                for j in range(len(rtn.tg_graph[i])):
                    if uniform(0.0,1.0) >= dropout:
                        def loss_with_reg_sampling(pid_params, target):
                            return add(samplinglossfunction(pid_params, target), Lam)
                        def loss_with_reg_static(x, y):
                            return add(staticlossfunction(x, y), Lam)
                        derivative = self.tg_graph_loss_derivative(i, j, ip, op, target_pid, loss_with_reg_sampling, loss_with_reg_static, rtn, maxt, sample_rate)
                        delta = mul(r, derivative)
                        if delta > maxdw:
                            delta = maxdw
                        if delta < -maxdw:
                            delta = -maxdw
                        rtn.tg_graph[i][j] = sub(rtn.tg_graph[i][j], delta)

        except Exception as error:
            rtn.tg = original_tg
            rtn.sr_graph = original_sr_graph
            rtn.tg_graph = original_tg_graph
            return error

        return True
    
    def save_training_data(self, data: dict, data_name: str = None, save_format: str = 'pkl') -> str:
        models_dir = self.get_models_dir()
        
        if data_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_name = f"training_data_{timestamp}"
        
        data_with_info = {
            'data': data,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_name': data_name,
                'format': save_format
            }
        }
        
        if save_format == 'pkl':
            file_path = models_dir / f"{data_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data_with_info, f)
        else:
            file_path = models_dir / f"{data_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_with_info, f, ensure_ascii=False, indent=2)
        
        return data_name

    def load_training_data(self, data_name: str, load_format: str = None):
        models_dir = self.get_models_dir()
        
        if load_format is None:
            if (models_dir / f"{data_name}.pkl").exists():
                load_format = 'pkl'
            elif (models_dir / f"{data_name}.json").exists():
                load_format = 'json'
            else:
                return None
        
        if load_format == 'pkl':
            file_path = models_dir / f"{data_name}.pkl"
            with open(file_path, 'rb') as f:
                data_with_info = pickle.load(f)
        else:
            file_path = models_dir / f"{data_name}.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data_with_info = json.load(f)
        
        return data_with_info.get('data', data_with_info)
    
    def save_rtn_state(self, rtn: RTN, model_name: str = None, save_format: str = 'pkl') -> str:
        models_dir = self.get_models_dir()
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"rtn_model_{timestamp}"
        
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, tuple) else k: convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        weights_converted = convert_keys(rtn.weights)
        
        model_state = {
            'weights': weights_converted,
            'tg': rtn.tg,
            'sr_graph': rtn.sr_graph,
            'tg_graph': rtn.tg_graph,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_name': model_name,
                'format': save_format
            }
        }
        
        if save_format == 'pkl':
            file_path = models_dir / f"{model_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(model_state, f)
        else:
            file_path = models_dir / f"{model_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(model_state, f, ensure_ascii=False, indent=2)
        
        info_path = models_dir / f"{model_name}_info.json"
        info_data = {
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'format': save_format,
            'model_type': 'RTN'
        }
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)
        
        return model_name

    def load_rtn_state(self, rtn: RTN, model_name: str, load_format: str = None) -> bool:
        models_dir = self.get_models_dir()
        
        info_path = models_dir / f"{model_name}_info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info_data = json.load(f)
                if 'format' in info_data:
                    load_format = info_data['format']
        
        if load_format is None:
            if (models_dir / f"{model_name}.pkl").exists():
                load_format = 'pkl'
            elif (models_dir / f"{model_name}.json").exists():
                load_format = 'json'
            else:
                return False
        
        if load_format == 'pkl':
            file_path = models_dir / f"{model_name}.pkl"
            with open(file_path, 'rb') as f:
                model_state = pickle.load(f)
        else:
            file_path = models_dir / f"{model_name}.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                model_state = json.load(f)
        
        def restore_keys(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    try:
                        if k.startswith('(') and k.endswith(')'):
                            new_key = eval(k)
                        else:
                            new_key = k
                    except:
                        new_key = k
                    result[new_key] = restore_keys(v)
                return result
            elif isinstance(obj, list):
                return [restore_keys(item) for item in obj]
            else:
                return obj
        
        rtn.weights = restore_keys(model_state['weights'])
        rtn.tg = model_state['tg']
        rtn.sr_graph = model_state['sr_graph']
        rtn.tg_graph = model_state['tg_graph']
        
        return True

    def save_training_pair(self, inputs: list, outputs: list, target_pids: list = None, pair_name: str = None) -> str:
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'target_pids': target_pids if target_pids else [],
            'pair_count': len(inputs)
        }
        return self.save_training_data(data, pair_name)

    def load_training_pair(self, pair_name: str):
        data = self.load_training_data(pair_name)
        if data:
            return {
                'inputs': data.get('inputs', []),
                'outputs': data.get('outputs', []),
                'target_pids': data.get('target_pids', [])
            }
        return None

    def list_saved_models(self):
        models_dir = self.get_models_dir()
        models = []
        
        for info_file in models_dir.glob("*_info.json"):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    model_file = models_dir / f"{info['model_name']}.{info.get('format', 'pkl')}"
                    if model_file.exists():
                        info['size'] = model_file.stat().st_size
                        info['file_exists'] = True
                    else:
                        info['file_exists'] = False
                    models.append(info)
            except Exception as e:
                print(f"Error reading {info_file}: {e}")
        
        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)

    def list_saved_data(self, data_type: str = 'all') -> list:
        models_dir = self.get_models_dir()
        files_info = []
        
        patterns = []
        if data_type in ['all', 'pkl']:
            patterns.append('*.pkl')
        if data_type in ['all', 'json']:
            patterns.append('*.json')
        
        for pattern in patterns:
            for file_path in models_dir.glob(pattern):
                if '_info' in file_path.stem:
                    continue
                
                file_info = {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'type': file_path.suffix[1:]
                }
                
                info_path = models_dir / f"{file_path.stem}_info.json"
                if info_path.exists():
                    with open(info_path, 'r', encoding='utf-8') as f:
                        file_info['info'] = json.load(f)
                
                files_info.append(file_info)
        
        return sorted(files_info, key=lambda x: x.get('modified', ''), reverse=True)

    def delete_saved_data(self, data_name: str) -> bool:
        models_dir = self.get_models_dir()
        deleted = False
        
        for ext in ['.pkl', '.json']:
            file_path = models_dir / f"{data_name}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            info_path = models_dir / f"{data_name}_info.json"
            if info_path.exists():
                info_path.unlink()
                deleted = True
        
        return deleted

if __name__ == "__main__":
    ipt = input(
"""Please choose a test content:
1) static trainer
2) PID controller
3) rtn_sampling trainer
4) complete training paradigm with data interface
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
        maxdw = 0.05
        dropout = 0.2
        
        print(f"\n{r = }")
        print(f"{maxdw = }")
        print(f"{dropout = }")

        print(f"\n{r = }\n")

        for epoch in range(50):
            print(f"## epoch {epoch+1}")

            print(t.static_trainer(ip, op, static_loss, lambda_loss, r, maxdw, dropout, rtn))

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
        print("# sampling trainer (single pair) start\n")

        t = RTN_Trainer()
        rtn = RTN(neurons_generator('any'),
                weights_brush(0.0, 0.1, 2), 
                tg_brush(), 
                sr_graph_brush(), 
                tg_graph_brush()
                )
        
        ip = [[1, 0, 0]]
        op = [[0, 0, 1]]
        tp = [[0.1, 0.01, 0.05]]
        
        print("data (single pair):")
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
        for i, o in zip(ip, op):
            sampling = t.rtn_sample(rtn, i, maxt, sample_rate)
            output_val = sampling[-1][0] if sampling and sampling[-1] else 0
            target_val = o[0] if o else 0
            print(f"input {i} -> output {output_val}, target {target_val}")
        
        print("\nstarting training on single pair...\n")
        
        for epoch in range(5):
            print(f"## epoch {epoch+1}")
            
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
            
            test_input = ip[0]
            test_output = rtn.nn_dynamics(test_input)[-1]
            test_val = test_output[0] if test_output else 0
            target_val = op[0][0] if op[0] else 0
            
            error = static_loss(test_output, op[0]) if test_output else 0
            
            print(f"error = {error}, target = {target_val}, output = {test_val}")
        
        print("\ntraining completed!\n")
        
        print("test on trained pair:")
        test_input = ip[0]
        test_output = rtn.nn_dynamics(test_input)[-1]
        test_val = test_output[0] if test_output else 0
        target_val = op[0][0] if op[0] else 0
        print(f"[{test_input[0]},0,0] -> {test_val} (target: {target_val})")
        
        print("\nfitted PID parameters:")
        sampling = t.rtn_sample(rtn, ip[0], maxt, sample_rate)
        pid_params = t.fitting_pid_for_(sampling)
        print(f"input {ip[0]} -> PID: {pid_params[0] if pid_params else 'N/A'}")
        
        print("\ntest on new input (should not be trained):")
        new_input = [2, 0, 0]
        new_output = rtn.nn_dynamics(new_input)[-1]
        new_val = new_output[0] if new_output else 0
        print(f"[2,0,0] -> {new_val}")

    elif ipt == "4":
        print("# Complete Training Paradigm with Data Interface\n")
        
        t = RTN_Trainer()
        
        print("Step 1: Try to Load Existing Model")
        print("-" * 40)
        
        available_models = t.list_saved_models()
        rtn = None
        
        if available_models:
            print("Found existing models:")
            for i, model in enumerate(available_models[:5]):
                print(f"  {i+1}. {model['model_name']} ({model['created_at'][:19]})")
            
            choice = input("Load existing model? (enter number or 'n' for new): ")
            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                model_idx = int(choice) - 1
                model_name = available_models[model_idx]['model_name']
                rtn = RTN(neurons_generator('any'), weights_brush(0.0, 0.1, 2), tg_brush(), sr_graph_brush(), tg_graph_brush())
                if t.load_rtn_state(rtn, model_name):
                    print(f"Loaded model: {model_name}")
                else:
                    rtn = None
        
        if rtn is None:
            print("Creating new model...")
            rtn = RTN(neurons_generator('any'),
                    weights_brush(0.0, 0.1, 2), 
                    tg_brush(), 
                    sr_graph_brush(), 
                    tg_graph_brush())
            print("New model created")
        
        print("\nStep 2: Try to Load Existing Training Data")
        print("-" * 40)
        
        all_data = t.list_saved_data()
        training_pairs = [d for d in all_data if d['name'] == 'training_pair']
        
        ip = None
        op = None
        tp = None
        
        if training_pairs:
            print("Found existing training data:")
            for data in training_pairs:
                print(f"  {data['name']}.{data['type']} ({data['size']} bytes) - {data['modified'][:19]}")
            
            load_data = input("Load existing training data? (y/n): ")
            if load_data.lower() == 'y':
                loaded = t.load_training_pair('training_pair')
                if loaded:
                    ip = loaded['inputs']
                    op = loaded['outputs']
                    tp = loaded['target_pids']
                    print(f"Loaded {len(ip)} training pairs")
        
        if ip is None:
            print("Creating training data...")
            ip = [[2*i+1,0,0] for i in range(4)] + [[0,2*i+1,0] for i in range(4)] + [[0,0,2*i+1] for i in range(4)]
            op = [[0,0,2*i+1] for i in range(4)] + [[0,0,0] for _ in range(8)]
            tp = [[0.1, 0.01, 0.05] for _ in range(len(ip))]
            
            print("Training pairs created:")
            for i,j in zip(ip[:5], op[:5]):
                print(f"  {i} -> {j}")
            print(f"  ... and {len(ip)-5} more pairs")
        
        print("\nStep 3: Configure Training Parameters")
        print("-" * 40)
        
        def static_loss(x, op):
            loss = 0.0
            for o, t in zip(x, op):
                diff = sub(o, t)
                loss = add(loss, mul(diff, diff))
            return loss
        
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
        
        def st_lambda_loss(weight):
            return mul(mul(st_r, 0.01), mul(weight, weight))
        
        def dy_lambda_loss(weight):
            return mul(mul(dy_r, 0.01), mul(weight, weight))
        
        st_r = 0.03
        st_maxdw = 0.05
        st_dropout = 0.2
        st_epochs = 10
        dy_r = 0.3
        dy_maxdw = 0.5
        dy_dropout = 0.2
        dy_epochs = 5
        dy_maxt = 30
        dy_sample_rate = 5
        
        print(f"Static training params: {st_r = }, {st_maxdw = }, {st_dropout = }, {st_epochs = }")
        print(f"Dynamic training params: {dy_r = }, {dy_maxdw = }, {dy_dropout = }, {dy_epochs = }, {dy_maxt = }, {dy_sample_rate = }")
        
        print("\nStep 4: Static Training")
        print("-" * 40)
        
        for epoch in range(st_epochs):
            print(f"\nStatic Epoch {epoch+1}/{st_epochs}")
            
            result = t.static_trainer(ip, op, static_loss, st_lambda_loss, st_r, st_maxdw, st_dropout, rtn)
            
            if isinstance(result, Exception):
                print(f"Training error: {result}")
                break
            
            total_error = 0.0
            for i, o in zip(ip, op):
                x = rtn.nn_dynamics(i)
                total_error = add(total_error, static_loss(x, o))
            
            print(f"  Static Loss: {total_error}")
            
            if epoch % 2 == 1:
                test_input = ip[0]
                test_output = rtn.nn_dynamics(test_input)[-1]
                print(f"  Sample output: {test_input} -> {test_output}")
        
        print("\nStep 5: Dynamic Training (Sampling-based)")
        print("-" * 40)
        
        single_ip = [ip[0]]
        single_op = [op[0]]
        single_tp = [tp[0]] if tp else [[0.1, 0.01, 0.05]]
        
        print(f"Training on single pair: {single_ip[0]} -> {single_op[0]}")
        print(f"Target PID: {single_tp[0]}")
        
        for epoch in range(dy_epochs):
            print(f"\nDynamic Epoch {epoch+1}/{dy_epochs}")
            
            result = t.rtn_sampling_trainer(
                single_ip, single_op, single_tp,
                sample_loss, static_loss, dy_lambda_loss,
                dy_r, dy_maxdw, dy_dropout, rtn,
                dy_maxt, dy_sample_rate
            )
            
            if isinstance(result, Exception):
                print(f"Dynamic training error: {result}")
                break
            
            sampling = t.rtn_sample(rtn, single_ip[0], dy_maxt, dy_sample_rate)
            pid_params = t.fitting_pid_for_(sampling, dy_maxt, dy_sample_rate)
            
            output_val = sampling[-1][0] if sampling and sampling[-1] else 0
            target_val = single_op[0][0]
            
            error = static_loss(sampling[-1] if sampling else [0], single_op[0])
            
            print(f"  Error: {error}, Output: {output_val}, Target: {target_val}")
            if pid_params:
                print(f"  Fitted PID: kp={pid_params[0][0]:.4f}, ki={pid_params[0][1]:.4f}, kd={pid_params[0][2]:.4f}")
        
        print("\nStep 6: Test Trained Model")
        print("-" * 40)
        
        print("Testing on training data:")
        for i in range(min(3, len(ip))):
            test_input = ip[i]
            test_output = rtn.nn_dynamics(test_input)[-1]
            target_output = op[i]
            print(f"  {test_input} -> {test_output} (target: {target_output})")
        
        print("\nTesting on new data:")
        new_inputs = [[5,0,0], [0,5,0], [0,0,5], [7,2,1]]
        for test_input in new_inputs:
            test_output = rtn.nn_dynamics(test_input)[-1]
            print(f"  {test_input} -> {test_output}")
        
        print("\nTesting sampling behavior:")
        for test_input in new_inputs[:2]:
            sampling = t.rtn_sample(rtn, test_input, dy_maxt, dy_sample_rate)
            pid_params = t.fitting_pid_for_(sampling, dy_maxt, dy_sample_rate)
            print(f"  Input {test_input}:")
            print(f"    Final output: {sampling[-1] if sampling else 'N/A'}")
            if pid_params:
                print(f"    PID params: kp={pid_params[0][0]:.4f}, ki={pid_params[0][1]:.4f}, kd={pid_params[0][2]:.4f}")
        
        print("\nStep 7: Save All Results")
        print("-" * 40)
        
        model_name = t.save_rtn_state(rtn, "trained_model_dynamic")
        print(f"Model saved as: {model_name}.pkl")
        print(f"Info file saved as: {model_name}_info.json")
        
        pair_name = t.save_training_pair(ip, op, tp, "training_pair")
        print(f"Training data saved as: {pair_name}.pkl")
        
        print("\nStep 8: View All Saved Files")
        print("-" * 40)
        
        all_files = t.list_saved_data()
        print(f"Total saved files: {len(all_files)}")
        for f in all_files:
            print(f"  {f['name']}.{f['type']} ({f['size']} bytes) - {f['modified'][:19]}")
            if 'info' in f:
                print(f"    Info: {f['info'].get('model_name', 'N/A')}")
        
        print("\nStep 9: Verify Loaded Model")
        print("-" * 40)
        
        verify_rtn = RTN(neurons_generator('any'),
                        weights_brush(0.0, 0.1, 2), 
                        tg_brush(), 
                        sr_graph_brush(), 
                        tg_graph_brush())
        
        if t.load_rtn_state(verify_rtn, model_name):
            print(f"Successfully loaded model: {model_name}")
            test_output = verify_rtn.nn_dynamics(ip[0])[-1]
            original_output = rtn.nn_dynamics(ip[0])[-1]
            print(f"Original output: {original_output}")
            print(f"Loaded output:   {test_output}")
            if original_output == test_output:
                print("Model verification passed!")
            else:
                print("Model verification failed!")
        
        print("\n" + "="*40)
        print("Complete Training Paradigm Finished!")
        print("="*40)

    else: print(None)