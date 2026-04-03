import sys
from pathlib import Path
from typing import List, Callable
from random import uniform
udir = str(Path(__file__).parent.parent.parent)
sys.path.append(udir)
from Model.CCATransformer import *
from Model.mathexpand import *
from Model.Vector import Vector
import math

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
            original_data = ccat.database
            bigQ = get_meaning_of_sentence_for_(ccat, tokens[:-1])
            
            perturbation = Vector([0.0] * dim)
            perturbation.components[i] = self.dx
            
            value = ccat.SoftQuery(tokens[-1], bigQ)

            ccat.SoftInjection_to_(tokens[-1], bigQ, sub(value, perturbation))
            sub_loss = loss(get_meaning_of_tokens_for_(ccat, tokens)[-1])
            ccat.database = original_data
            
            ccat.SoftInjection_to_(tokens[-1], bigQ, add(value, perturbation))
            add_loss = loss(get_meaning_of_tokens_for_(ccat, tokens)[-1])
            ccat.database = original_data
            
            gradient.components[i] = (add_loss - sub_loss) / (2 * self.dx)
        
        return gradient

if __name__ == '__main__':
    string = """父亲的旧手表
父亲的抽屉里，有一只不走的旧手表。
表盘已经发黄，时针永远停在三点十七分。我无数次想过要把它拿去修好，父亲总是摆摆手：“算了，让它歇着吧。”
这只表是父亲二十五岁时买的。那时他在建筑工地打工，每天凌晨四点起床，赶第一班公交车去工地。他说，有次手表慢了五分钟，他迟到了，被扣了半天工资。从那以后，他每天都要对着新闻联播校准时间。
我上小学那年，父亲用这只表教我认时间。我总也分不清时针和分针，他就把手表戴在我细小的手腕上，一圈一圈地转着给我讲解。“你看，短的是时针，像乌龟，走得慢；长的是分针，像兔子，跑得快。”阳光透过窗户，照在表盘上，折射出细碎的光芒。
高二那年冬天的一个深夜，我突然发高烧。父亲从睡梦中惊醒，抓起手表看了一眼，背起我就往医院跑。雪花飘落在他花白的头发上，他的呼吸在寒风中结成白雾。后来我才知道，那晚他太着急，手表磕在门框上，摔坏了。
高考前夜，我紧张得睡不着。父亲来到我床边，把那只不走的表放在我枕边：“别怕，时间不会停，你只管往前走。”
如今，我即将步入大学。收拾行李时，父亲突然把那只旧手表塞进我的书包：“带上吧，它虽然不走了，但它陪着我走过了最好的年华。现在，让它陪着你。”
我把手表贴在耳边，仿佛听见了时间的回声——那是父亲的青春，也是我的起点。三点十七分，时间定格，而爱，永远向前。"""

    from Data.splitToken import *
    ct  = ChineseTokenizer()
    ct.load_model("model")
    text = ct.tokenize(string)

    c = NewCCATransformer(text)
    for i in range(9): c = FusionCCATransformer(c , NewCCATransformer(text))
    c.temperature = 0.6

    t = CCAT_Trainer(dx=1e-5)

    def l(x):
        def softmax(p):
            return math.exp(p)/(math.exp(i)+math.exp(j))
        
        hartley = []

        for i,j in zip(x.components, y.components):
            hartley.append(softmax(j) * math.log(softmax(i)) + (1-softmax(j)) * math.log(1-softmax(i)))

        return sum(hartley)

    cnt = 0
    while cnt < 50:
        cnt += 1
        print(f"{cnt = }")

        for i in [text[:j] for j in range(3, len(text))]:
            y = get_meaning_of_tokens_for_(c, i)[-1]
            if uniform(0.0, 1.0) >= 0.3:
                garadient = t.loss_gradient(i[:-1], l, c)
                print(f"token = {i[-1]}:\nbigQ = {y}, {garadient = }")
                c.SoftInjection_to_(i[-1], y, sub(c.SoftQuery(i[-1], y), mul(garadient, 0.03)))

    tokens = text[:2]
    print(tokens)
    while True:
        next_token = softmax_choice_next_token_for_(c, tokens)
        print(next_token)
        tokens.append(next_token)