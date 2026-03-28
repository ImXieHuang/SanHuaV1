import sys
from pathlib import Path
from typing import List, Callable
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)
from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import *
import Model.mathexpand as mexp
from Model.Vector.vector import Vector

class CCAT_Trainer:

    def __init__(self, dx: float=None):
        self.dx = dx or 0.001
        self.models_dir = self.get_models_dir()

    def get_models_dir(self) -> Path:
        current_dir = Path(__file__).parent
        models_dir = current_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        return models_dir

    def loss_gradient(self, tokens: List[str], loss: Callable, ccat: CCATransformer):
        pass

if __name__ == '__main__':
    text = ['你好', '我', '是', '三花']
    c = NewCCATransformer(text)
    for i in range(3): c = FusionCCATransformer(c , NewCCATransformer(text))
    t = CCAT_Trainer(dx=0.01)