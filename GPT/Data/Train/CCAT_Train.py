import sys
import copy
from pathlib import Path
from typing import List, Callable
from datetime import datetime
import json
import pickle
import random as rand
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)
from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import Vector
from math import log2

class CCAT_Trainer:
    def __init__(self, dx: float=None):
        self.dx = dx or 0.001
        self.models_dir = self.get_models_dir()

    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, tokens: List[str], loss: Callable, ccat: CCATransformer) -> Vector:
        dim = ccat.dim
        gradient = Vector([0.0] * dim)
        
        for i in range(dim):
            original_data = copy.deepcopy(ccat.database)
            bigQ = get_meaning_of_sentence_for_(ccat, tokens[:-1])
            
            perturbation = Vector([0.0] * dim)
            perturbation.components[i] = self.dx
            
            value = ccat.SoftQuery(tokens[-1], bigQ)

            ccat.SoftInjection_to_(tokens[-1], bigQ, sub(value, perturbation))
            sub_loss = loss(tokens, ccat)
            ccat.database = copy.deepcopy(original_data)
            
            ccat.SoftInjection_to_(tokens[-1], bigQ, add(value, perturbation))
            add_loss = loss(tokens, ccat)
            ccat.database = copy.deepcopy(original_data)
            
            gradient.components[i] = (add_loss - sub_loss) / (2 * self.dx)
        
        return gradient
    
    def trainer(self, tokens: List[str], lossfunction: callable, lambdafunction: callable, r: float, maxdw: float, dropout: float, ccat: CCATransformer):
        original_data = copy.deepcopy(ccat.database)
        try:
            def loss(token, ccat) -> float:
                Lam = 0.0
                Lambdacnt = 0
                for i in list(ccat.database.values()):
                    for j,k in list(i.items()):
                        Lam = add(add(Lam, lambdafunction(j)), lambdafunction(k))
                        Lambdacnt += 1
                Lam = sum(Lam.components)/Lambdacnt if Lambdacnt > 0 else 0.0

                return lossfunction(token, ccat) + Lam

            injection = {}

            for n_gram in [tokens[:i] for i in range(2, len(tokens))]:
                if rand.random() < dropout:
                    continue
                print(f"Training on {n_gram[:-1]} -> {n_gram[-1]}")
                gradient = self.loss_gradient(n_gram, loss, ccat)
                bigQ = get_meaning_of_sentence_for_(ccat, n_gram[:-1])
                injection[tuple(n_gram)] = (gradient, bigQ)
                print("Probabilities:")
                for token, prob in zip(*softmax_choice_next_probability_for_(ccat, n_gram[:-1])):
                    print(f"  {token}: {prob:.4f}")
                print(f"Loss: {loss(n_gram, ccat):.4f}\n")

            for token, data in injection.items():
                token = list(token)
                grad = data[0]
                bigQ = data[1]

                if grad.magnitude() > maxdw:
                    grad = grad.normalize() * maxdw
                value = ccat.SoftQuery(token[-1], bigQ)

                ccat.SoftInjection_to_(token[-1], bigQ, sub(value, grad))
                ccat.SoftInjection_query_to_(token[-1], bigQ)

        except Exception as error:
            ccat.database = copy.deepcopy(original_data)
            return error

        return True

    def cross_entropy(self, tokens: List[str], actuality: str, ccat: CCATransformer) -> float:
        p = softmax_choice_next_probability_for_(ccat, tokens)
        if actuality not in p[0]:
            return float('inf')
        return -log2(p[1][p[0].index(actuality)])

    def save_model(self, data: dict, data_name: str = None, save_format: str = 'pkl') -> str:
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
    
    def save_training_result(self, data: dict, data_name: str = None, save_format: str = 'pkl') -> str:
        data_name = self.save_model(data, data_name, save_format)
        info = {
            'model_name': data_name,
            'format': save_format,
            'created_at': datetime.now().isoformat()
        }
        info_path = self.get_models_dir() / f"{data_name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return data_name
        
    def load_model(self, data_name: str, load_format: str = None):
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
    
    def load_training_result(self, data_name: str, load_format: str = None):
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
    from Data.splitToken import ChineseTokenizer
    from Model.mathexpand import iterate
    from time import time
    import sys

    log_file = Path(__file__).parent / "training_log.txt"
    sys.stdout = open(log_file, "w")

    t = CCAT_Trainer()
    
    start_time = time()

    print("Loading and preparing data...")

    text = [
        "我常想在纷扰中寻出一点闲静来，",
        "然而委实不容易。",
        "目前是这么离奇，",
        "心里是这么芜杂。",
        "一个人做到只剩了回忆的时候，",
        "生涯大概总要算是无聊了罢，",
        "但有时竟会连回忆也没有。",
        "中国的做文章有轨范，",
        "世事也仍然是螺旋。",
        "不容易。目前",
        "离奇，心里是",
        "芜杂。一个人",
        "的时候，生涯",
        "了罢，但",
        "也没有。中国",
        "轨范，世事"
    ]

    ct = ChineseTokenizer()
    ct.load_model("model")

    Texts = [["<START>"] + ct.tokenize(i) + ["<END>"]*2 for i in text]

    print(f"Data loaded and prepared: {Texts}")

    print(f"Data preparation completed in {time() - start_time:.2f} seconds.\n")

    sys.stdout.flush()

    available_models = t.list_saved_models()
    if available_models:
        print("Available saved models:")
        for model in available_models:
            print(f"  - {model['model_name']} (created at {model['created_at']}, size: {model.get('size', 'unknown')} bytes, file exists: {model.get('file_exists', False)})")

        choice = input("Load existing model? (enter number or 'n' for new): ")

        if choice.isdigit() and 1 <= int(choice) <= len(available_models):
            selected_model = available_models[int(choice) - 1]
            print(f"Loading model: {selected_model['model_name']}...")
            ccat_data = t.load_training_result(selected_model['model_name'], selected_model.get('format', 'pkl'))
            if ccat_data is not None:
                ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(list({i for j in Texts for i in j}), dim = 2) for _ in range(4)])
                ccat.database = ccat_data
                print("Model loaded successfully.")
            else:
                print("Failed to load model data. Starting with a new model.")
                ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(list({i for j in Texts for i in j}), dim = 2) for _ in range(4)])
        else:
            print("Starting with a new model.")
            ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(list({i for j in Texts for i in j}), dim = 2) for _ in range(4)])
    else:
        print("No saved models found. Starting with a new model.")
        ccat: CCATransformer = iterate(FusionCCATransformer, [NewCCATransformer(list({i for j in Texts for i in j}), dim = 2) for _ in range(4)])

    print()

    ccat.temperature = 1.0

    training_start_time = time()

    start_time = time()

    print("Starting training...")

    r = 0.75

    for cnt in range(10):
        start_time = time()

        print(f"Epoch {cnt + 1}")

        for Text in Texts:  
            print(f"Training on {Text} are {t.trainer(Text, lambda tokens, ccat: t.cross_entropy(tokens, Text[Text.index(tokens[-1]) + 1] if len(tokens) > 1 else 0.0, ccat)  + sum([i[0]**4 if i[1] in ['<START>'] else 0.0 for i in softmax_choice_next_probability_for_(ccat, tokens)]), lambda x: 0.01 * x**2, r, 2.5, 0.1, ccat)}")
            print()

        print(f"Epoch {cnt + 1} completed in {time() - start_time:.2f} seconds.\n")

        print("test:")

        for i in range(len(Texts)):
            print(f"Sequence: {''.join(Texts[i])}")
            for j in range(len(Texts[i]) - 1):
                print(f"{'……' if j >= 1 else '  '} {Texts[i][j]} -> {Texts[i][j + 1]}:   {softmax_choice_next_token_for_(ccat, Texts[i][:j])}")
            print()

        print()

        sys.stdout.flush()

    print("Training completed.")

    print()

    print(f"Total training time: {time() - training_start_time:.2f} seconds.\n")

    print("Save model...")

    model_name = t.save_training_result(ccat.database, "ccat_model", "pkl")

    print(f"Model saved as {model_name}.")

    print()

    print("Testing the model 10 times...")

    for i in range(10):
        tokens = ["这"]
        while tokens[-1] != "<END>" and len(tokens) < 20:
            next_token = softmax_choice_next_token_for_(ccat, tokens)
            tokens.append(next_token)
        print(f"Generated sequence {i + 1}: {''.join(tokens)}")