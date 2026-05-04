import sys
from pathlib import Path
from random import uniform

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from parrot import *
from pinyin import *

class speak:
    def __init__(self):
        self.phonemes = self.get_phonemes()
        self.pm = PinyinManager()

    def get_phonemes(self):
        phonemes = {}
        
        phonemes['a'] = morph_key([2.0, 1.5, 1.0, 0.8, 0.8, 1.0, 1.5, 2.0], 0.0, 1.0)
        phonemes['o'] = morph_key([1.0, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0], 0.0, 1.0)
        phonemes['e'] = morph_key([1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0], 0.0, 1.0)
        phonemes['i'] = morph_key([0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8], 0.0, 1.0)
        phonemes['u'] = morph_key([0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.6, 0.3], 0.0, 1.0)
        phonemes['ü'] = morph_key([0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.5, 0.4], 0.0, 1.0)
        phonemes['ê'] = morph_key([0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0, 1.0)
        phonemes['er'] = morph_key([0.6, 0.8, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9], 0.0, 1.0)
        phonemes['ai'] = morph_key([1.8, 1.5, 1.0, 0.6, 0.4, 0.4, 0.4, 0.4], 0.0, 1.0)
        phonemes['ei'] = morph_key([1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.3, 0.3], 0.0, 1.0)
        phonemes['ao'] = morph_key([1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.9, 0.9], 0.0, 1.0)
        phonemes['ou'] = morph_key([1.0, 1.1, 1.2, 1.0, 0.8, 0.5, 0.4, 0.4], 0.0, 1.0)
        phonemes['ia'] = morph_key([0.4, 0.6, 0.8, 1.1, 1.4, 1.7, 1.9, 1.5], 0.0, 1.0)
        phonemes['ie'] = morph_key([0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.1, 1.1], 0.0, 1.0)
        phonemes['ua'] = morph_key([0.6, 0.8, 1.0, 1.3, 1.5, 1.4, 1.2, 1.0], 0.0, 1.0)
        phonemes['uo'] = morph_key([0.7, 0.9, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0], 0.0, 1.0)
        phonemes['üe'] = morph_key([0.4, 0.6, 0.8, 1.0, 1.2, 1.2, 1.2, 1.2], 0.0, 1.0)
        phonemes['iu'] = morph_key([0.4, 0.7, 1.0, 0.9, 0.6, 0.3, 0.3, 0.3], 0.0, 1.0)
        phonemes['iao'] = morph_key([0.4, 0.7, 1.0, 1.4, 1.5, 1.2, 0.8, 0.8], 0.0, 1.0)
        phonemes['iou'] = morph_key([0.4, 0.7, 1.0, 0.9, 0.6, 0.3, 0.3, 0.3], 0.0, 1.0)
        phonemes['uai'] = morph_key([0.7, 1.0, 1.3, 1.3, 1.0, 0.6, 0.3, 0.3], 0.0, 1.0)
        phonemes['uei'] = morph_key([0.6, 0.9, 1.2, 1.0, 0.7, 0.4, 0.2, 0.2], 0.0, 1.0)
        phonemes['an'] = morph_key([1.8, 1.5, 1.0, 0.7, 0.6, 0.6, 0.6, 0.6], 0.3, 1.0)
        phonemes['en'] = morph_key([1.2, 0.9, 0.6, 0.5, 0.4, 0.4, 0.4, 0.4], 0.3, 1.0)
        phonemes['in'] = morph_key([0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5], 0.4, 1.0)
        phonemes['un'] = morph_key([0.8, 1.0, 1.0, 0.8, 0.6, 0.5, 0.5, 0.5], 0.3, 1.0)
        phonemes['ün'] = morph_key([0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7], 0.4, 1.0)
        phonemes['ian'] = morph_key([0.3, 0.6, 1.0, 1.2, 1.1, 0.9, 0.9, 0.9], 0.3, 1.0)
        phonemes['uan'] = morph_key([0.6, 0.9, 1.2, 1.1, 0.8, 0.7, 0.7, 0.7], 0.3, 1.0)
        phonemes['üan'] = morph_key([0.4, 0.7, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8], 0.4, 1.0)
        phonemes['ang'] = morph_key([1.8, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], 0.6, 1.0)
        phonemes['eng'] = morph_key([1.2, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], 0.6, 1.0)
        phonemes['ing'] = morph_key([0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6], 0.7, 1.0)
        phonemes['ong'] = morph_key([0.7, 1.0, 1.2, 1.0, 0.7, 0.7, 0.7, 0.7], 0.5, 1.0)
        phonemes['iang'] = morph_key([0.4, 0.7, 1.1, 1.4, 1.3, 1.0, 0.9, 0.9], 0.6, 1.0)
        phonemes['uang'] = morph_key([0.6, 0.9, 1.2, 1.3, 1.1, 0.8, 0.8, 0.8], 0.6, 1.0)
        phonemes['iong'] = morph_key([0.4, 0.7, 1.0, 1.0, 0.8, 0.6, 0.6, 0.6], 0.5, 1.0)
        phonemes['ueng'] = morph_key([0.9, 1.2, 1.3, 1.1, 0.8, 0.8, 0.8, 0.8], 0.5, 1.0)
        phonemes['b'] = morph_key([0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0], 0.0, 1.0)
        phonemes['p'] = morph_key([0.0, 0.0, 0.0, 0.2, 0.5, 0.7, 0.9, 0.9], 0.0, 0.0)
        phonemes['m'] = morph_key([0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 0.8, 0.8], 0.9, 1.0)
        phonemes['f'] = morph_key([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.9, 0.9], 0.0, 0.0)
        phonemes['d'] = morph_key([0.2, 0.1, 0.0, 0.0, 0.2, 0.5, 0.8, 0.8], 0.0, 1.0)
        phonemes['t'] = morph_key([0.2, 0.1, 0.0, 0.0, 0.2, 0.5, 0.7, 0.7], 0.0, 0.0)
        phonemes['n'] = morph_key([0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.7, 0.7], 0.9, 1.0)
        phonemes['l'] = morph_key([0.3, 0.5, 0.6, 0.5, 0.6, 0.6, 0.7, 0.7], 0.0, 1.0)
        phonemes['g'] = morph_key([0.8, 0.5, 0.2, 0.0, 0.0, 0.2, 0.5, 0.6], 0.0, 1.0)
        phonemes['k'] = morph_key([0.8, 0.5, 0.2, 0.0, 0.0, 0.2, 0.4, 0.5], 0.0, 0.0)
        phonemes['h'] = morph_key([0.9, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2, 0.2], 0.0, 0.0)
        phonemes['j'] = morph_key([0.3, 0.2, 0.0, 0.1, 0.3, 0.5, 0.6, 0.6], 0.0, 1.0)
        phonemes['q'] = morph_key([0.3, 0.2, 0.0, 0.1, 0.3, 0.5, 0.5, 0.5], 0.0, 0.0)
        phonemes['x'] = morph_key([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2], 0.0, 0.0)
        phonemes['zh'] = morph_key([0.2, 0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.7], 0.0, 1.0)
        phonemes['ch'] = morph_key([0.2, 0.1, 0.0, 0.1, 0.3, 0.5, 0.6, 0.6], 0.0, 0.0)
        phonemes['sh'] = morph_key([0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3], 0.0, 0.0)
        phonemes['r'] = morph_key([0.4, 0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 0.5], 0.0, 1.0)
        phonemes['z'] = morph_key([0.1, 0.0, 0.0, 0.2, 0.4, 0.7, 0.8, 0.8], 0.0, 1.0)
        phonemes['c'] = morph_key([0.1, 0.0, 0.0, 0.2, 0.4, 0.6, 0.7, 0.7], 0.0, 0.0)
        phonemes['s'] = morph_key([0.2, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4], 0.0, 0.0)
        phonemes['y'] = morph_key([0.2, 0.1, 0.2, 0.4, 0.5, 0.6, 0.6, 0.6], 0.0, 1.0)
        phonemes['w'] = morph_key([0.6, 0.8, 0.7, 0.5, 0.3, 0.2, 0.2, 0.2], 0.0, 1.0)
        phonemes['ng'] = morph_key([0.8, 0.6, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2], 0.9, 1.0)

        return phonemes

    def callmorph(self, string):
        self.pm.loadpinyin()
        phonemes = self.pm.callstring(string)
        output = []
        
        for i in phonemes:
            if i != "[hangup]":
                if i[0] != "":
                    output.append((self.phonemes[i[0]], self.phonemes[i[1]]))
                else:
                    output.append((self.phonemes[i[1]], self.phonemes[i[1]]))
            else:
                output.append("[hangup]")

        return output
    
    def tts(self, text):
        p = parrot(12, 8, 0.33)
        
        for name, key in self.phonemes.items():
            if name is not None:
                p.add_key(name, key)
        
        ms = self.callmorph(text)
        
        duration_per_phoneme = 0.15
        total_duration = len(ms) * duration_per_phoneme
        
        def speak_text(t):
            if t >= total_duration:
                return 0.0
            
            idx = int(t / duration_per_phoneme)
            if idx >= len(ms):
                return 0.0
            
            t_local = t - idx * duration_per_phoneme
            if ms[idx] != "[hangup]":
                key1, key2 = ms[idx]
                
                ratio = t_local / duration_per_phoneme
                
                areas1, nasal1, voiced1 = key1
                areas2, nasal2, voiced2 = key2
                
                areas = [areas1[i] * (1 - ratio) + areas2[i] * ratio for i in range(8)]
                nasal = nasal1 * (1 - ratio) + nasal2 * ratio
                voiced = voiced1 * (1 - ratio) + voiced2 * ratio
                
                p.get_key(morph_key(areas, nasal, voiced))
            
                return p.gate(*p.speak())(t) * p.s.envelope_adsr(t_local, duration_per_phoneme, 0.05, 0.1, 0.8, 0.1)
            
            return 0

        print(f"play: {text}")
        p.s.play_func(speak_text, total_duration)

if __name__ == "__main__":
    s = speak()
    text = "我是三花"

    s.tts(text)