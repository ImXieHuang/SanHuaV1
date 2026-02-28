import sys
from pathlib import Path

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from GPT.Modle.Remember_Turingpattern_NN import RTN, neurons_generator, weights_brush, tg_brush, sr_graph_brush, tg_graph_brush
from GPT.Modle.CCATransformer import CCATransformer, Const
from GPT.Modle.Vector import Vector

cat = CCATransformer(
        {
            "苹果": {
                Vector([1,0,0,0]): Vector([1,1,1,1]),
                Vector([0,0,1,0]): Vector([2,2,2,2])
            },
            "香蕉": {
                Vector([0,1,0,0]): Vector([2,2,2,2])
            }
        },
        temperature=Const.E
    )
