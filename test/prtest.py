import sys
from pathlib import Path
from random import uniform

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from Outputio.parrot import *

# 中文音素表设计 - 基于鹦鹉合成器参数
class ChinesePhonemeTable:
    def __init__(self, n_segments=4):
        self.n = n_segments
        
    def get_phonemes(self):
        """返回所有中文音素"""
        phonemes = {}
        
        # ========== 单元音 ==========
        # 开口元音 - 口腔开放，面积大
        phonemes['a'] = morph_key([2.0, 1.8, 1.5, 1.2], 0.0, 1.0)  # 啊
        phonemes['o'] = morph_key([1.0, 0.8, 1.2, 1.5], 0.0, 1.0)  # 哦 - 圆唇
        phonemes['e'] = morph_key([1.2, 1.0, 0.8, 1.0], 0.0, 1.0)  # 鹅 - 中部略窄
        phonemes['i'] = morph_key([0.3, 0.2, 0.4, 0.8], 0.0, 1.0)  # 衣 - 前部狭窄
        phonemes['u'] = morph_key([0.8, 1.2, 0.6, 0.3], 0.0, 1.0)  # 乌 - 后部狭窄，圆唇
        phonemes['ü'] = morph_key([0.3, 0.4, 0.5, 0.6], 0.0, 1.0)  # 迂 - i和u的混合
        
        # ========== 复合元音 ==========
        phonemes['ai'] = morph_key([1.8, 1.4, 0.8, 0.4], 0.0, 1.0)  # 爱 - a到i的过渡形态
        phonemes['ei'] = morph_key([1.2, 1.0, 0.6, 0.3], 0.0, 1.0)  # 诶 - e到i
        phonemes['ao'] = morph_key([1.8, 1.5, 1.2, 0.9], 0.0, 1.0)  # 奥 - a到o
        phonemes['ou'] = morph_key([1.0, 1.2, 0.8, 0.4], 0.0, 1.0)  # 欧 - o到u
        phonemes['ia'] = morph_key([0.4, 0.8, 1.4, 1.6], 0.0, 1.0)  # 呀 - i到a
        phonemes['ie'] = morph_key([0.3, 0.5, 0.9, 1.1], 0.0, 1.0)  # 耶 - i到e
        phonemes['ua'] = morph_key([0.6, 1.0, 1.5, 1.3], 0.0, 1.0)  # 哇 - u到a
        phonemes['uo'] = morph_key([0.7, 0.9, 1.2, 1.0], 0.0, 1.0)  # 我 - u到o
        phonemes['üe'] = morph_key([0.4, 0.6, 1.0, 1.2], 0.0, 1.0)  # 约 - ü到e
        
        # ========== 三合元音 ==========
        phonemes['iao'] = morph_key([0.4, 1.2, 1.4, 0.8], 0.0, 1.0)  # 要 - i-a-o
        phonemes['iou'] = morph_key([0.4, 1.0, 0.8, 0.3], 0.0, 1.0)  # 有 - i-o-u
        phonemes['uai'] = morph_key([0.7, 1.4, 0.8, 0.3], 0.0, 1.0)  # 外 - u-a-i
        phonemes['uei'] = morph_key([0.6, 1.2, 0.6, 0.2], 0.0, 1.0)  # 为 - u-e-i
        
        # ========== 鼻音韵母 ==========
        # 前鼻音 - 轻度鼻腔耦合
        phonemes['an'] = morph_key([1.8, 1.4, 1.0, 0.6], 0.3, 1.0)  # 安
        phonemes['en'] = morph_key([1.2, 0.9, 0.6, 0.4], 0.3, 1.0)  # 恩
        phonemes['in'] = morph_key([0.3, 0.3, 0.4, 0.5], 0.4, 1.0)  # 因
        phonemes['un'] = morph_key([0.8, 1.0, 0.7, 0.5], 0.3, 1.0)  # 温
        phonemes['ün'] = morph_key([0.4, 0.5, 0.6, 0.7], 0.4, 1.0)  # 晕
        
        # 后鼻音 - 较强鼻腔耦合
        phonemes['ang'] = morph_key([1.8, 1.4, 1.2, 0.8], 0.6, 1.0)  # 昂
        phonemes['eng'] = morph_key([1.2, 1.0, 0.8, 0.6], 0.6, 1.0)  # 鞥
        phonemes['ing'] = morph_key([0.3, 0.4, 0.5, 0.6], 0.7, 1.0)  # 英
        phonemes['ong'] = morph_key([0.7, 1.2, 1.0, 0.7], 0.5, 1.0)  # 翁
        
        # ========== 塞音 (爆破音) ==========
        # 不送气清音 - 完全闭塞后爆破
        phonemes['b'] = morph_key([0.0, 0.0, 0.5, 1.0], 0.0, 1.0)   # 波 - 双唇爆破
        phonemes['d'] = morph_key([0.2, 0.0, 0.0, 0.8], 0.0, 1.0)   # 得 - 舌尖爆破
        phonemes['g'] = morph_key([0.8, 0.0, 0.0, 0.6], 0.0, 1.0)   # 哥 - 舌根爆破
        
        # 送气清音 - 爆破后有气流
        phonemes['p'] = morph_key([0.0, 0.0, 0.4, 0.9], 0.0, 0.8)   # 坡 - 送气较强
        phonemes['t'] = morph_key([0.2, 0.0, 0.0, 0.7], 0.0, 0.8)   # 特 - 送气较强
        phonemes['k'] = morph_key([0.8, 0.0, 0.0, 0.5], 0.0, 0.8)   # 科 - 送气较强
        
        # ========== 塞擦音 ==========
        # 不送气塞擦音
        phonemes['z'] = morph_key([0.1, 0.0, 0.3, 0.8], 0.0, 0.7)   # 资 - 舌尖前
        phonemes['zh'] = morph_key([0.2, 0.0, 0.2, 0.7], 0.0, 0.7)  # 知 - 舌尖后
        phonemes['j'] = morph_key([0.3, 0.0, 0.1, 0.6], 0.0, 0.7)   # 基 - 舌面
        
        # 送气塞擦音
        phonemes['c'] = morph_key([0.1, 0.0, 0.2, 0.7], 0.0, 0.6)   # 雌 - 送气
        phonemes['ch'] = morph_key([0.2, 0.0, 0.1, 0.6], 0.0, 0.6)  # 吃 - 送气
        phonemes['q'] = morph_key([0.3, 0.0, 0.0, 0.5], 0.0, 0.6)   # 七 - 送气
        
        # ========== 擦音 ==========
        # 清擦音 - 声道狭窄，产生湍流
        phonemes['f'] = morph_key([0.0, 0.3, 0.5, 0.9], 0.0, 0.0)   # 佛 - 唇齿摩擦
        phonemes['s'] = morph_key([0.1, 0.2, 0.4, 0.8], 0.0, 0.0)   # 思 - 舌尖前
        phonemes['sh'] = morph_key([0.2, 0.2, 0.3, 0.7], 0.0, 0.0)  # 师 - 舌尖后
        phonemes['x'] = morph_key([0.3, 0.3, 0.2, 0.6], 0.0, 0.0)   # 西 - 舌面
        phonemes['h'] = morph_key([0.9, 0.7, 0.4, 0.2], 0.0, 0.0)   # 喝 - 喉部摩擦
        
        # 浊擦音 - 声带振动
        phonemes['r'] = morph_key([0.4, 0.3, 0.2, 0.5], 0.0, 0.9)   # 日 - 卷舌浊擦
        
        # ========== 鼻音 ==========
        phonemes['m'] = morph_key([0.0, 0.0, 0.4, 0.8], 0.9, 1.0)   # 摸 - 双唇鼻音
        phonemes['n'] = morph_key([0.2, 0.0, 0.2, 0.7], 0.9, 1.0)   # 讷 - 舌尖鼻音
        phonemes['ng'] = morph_key([0.8, 0.5, 0.3, 0.2], 0.8, 1.0)  # 嗯 - 舌根鼻音
        
        # ========== 边音 ==========
        phonemes['l'] = morph_key([0.3, 0.6, 0.5, 0.7], 0.0, 1.0)   # 勒 - 舌尖边音
        
        return phonemes
    
    def get_example_words(self):
        """返回示例词语的音素序列"""
        examples = {
            '妈妈': ['m', 'a', 'm', 'a'],
            '爸爸': ['b', 'a', 'b', 'a'],
            '中国': ['zh', 'ong', 'g', 'uo'],
            '你好': ['n', 'i', 'h', 'ao'],
            '老师': ['l', 'ao', 'sh', 'i'],
            '学习': ['x', 'üe', 'x', 'i'],
            '电脑': ['d', 'ian', 'n', 'ao'],
            '手机': ['sh', 'ou', 'j', 'i'],
            '天气': ['t', 'ian', 'q', 'i'],
            '音乐': ['y', 'in', 'y', 'üe'],
        }
        return examples

# 在鹦鹉合成器中使用
if __name__ == "__main__":
    p = parrot(12, 4)  # 15cm声道长度，4个分段
    
    # 加载音素表
    pt = ChinesePhonemeTable()
    phonemes = pt.get_phonemes()
    
    # 注册所有音素
    for name, key in phonemes.items():
        p.add_key(name, key)
    
    def say_miao(t):
        duration_total = 2.0
        duration_per = 1.0
        
        if t < duration_per:
            return p.gate(*p.get_key(p.syllable("m", "iao", t, duration_per)).speak())(t) * \
                   p.s.envelope_adsr(t, duration_per, 0.05, 0.1, 0.8, 0.1)
        else:
            t2 = t - duration_per
            if t2 < duration_per:
                return p.gate(*p.get_key(p.syllable("m", "iao", t2, duration_per)).speak())(t) * \
                       p.s.envelope_adsr(t2, duration_per, 0.05, 0.1, 0.7, 0.1)
            else:
                return 0.0
    
    print("播放'喵喵'")
    p.s.play_func(say_miao, 2.0)  # 明确指定2秒时长