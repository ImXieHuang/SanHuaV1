import sys
from pathlib import Path
from random import uniform

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from Outputio.parrot import *
from Outputio.pinyin import *

class speak:
    def __init__(self):
        self.phonemes = self.get_phonemes()
        self.pm = PinyinManager()

    def get_phonemes(self):
        phonemes = {}

        phonemes['a'] = morph_key([2.0, 1.8, 1.5, 1.2], 0.0, 1.0)
        phonemes['o'] = morph_key([1.0, 0.8, 1.2, 1.5], 0.0, 1.0)
        phonemes['e'] = morph_key([1.2, 1.0, 0.8, 1.0], 0.0, 1.0)
        phonemes['i'] = morph_key([0.3, 0.2, 0.4, 0.8], 0.0, 1.0)
        phonemes['u'] = morph_key([0.8, 1.2, 0.6, 0.3], 0.0, 1.0)
        phonemes['ü'] = morph_key([0.3, 0.4, 0.5, 0.6], 0.0, 1.0)
        phonemes['ai'] = morph_key([1.8, 1.4, 0.8, 0.4], 0.0, 1.0)
        phonemes['ei'] = morph_key([1.2, 1.0, 0.6, 0.3], 0.0, 1.0)
        phonemes['ao'] = morph_key([1.8, 1.5, 1.2, 0.9], 0.0, 1.0)
        phonemes['ou'] = morph_key([1.0, 1.2, 0.8, 0.4], 0.0, 1.0)
        phonemes['ia'] = morph_key([0.4, 0.8, 1.4, 1.6], 0.0, 1.0)
        phonemes['ie'] = morph_key([0.3, 0.5, 0.9, 1.1], 0.0, 1.0)
        phonemes['ua'] = morph_key([0.6, 1.0, 1.5, 1.3], 0.0, 1.0)
        phonemes['uo'] = morph_key([0.7, 0.9, 1.2, 1.0], 0.0, 1.0)
        phonemes['üe'] = morph_key([0.4, 0.6, 1.0, 1.2], 0.0, 1.0)
        phonemes['iao'] = morph_key([0.4, 1.2, 1.4, 0.8], 0.0, 1.0)
        phonemes['iou'] = morph_key([0.4, 1.0, 0.8, 0.3], 0.0, 1.0)
        phonemes['uai'] = morph_key([0.7, 1.4, 0.8, 0.3], 0.0, 1.0)
        phonemes['uei'] = morph_key([0.6, 1.2, 0.6, 0.2], 0.0, 1.0)
        phonemes['an'] = morph_key([1.8, 1.4, 1.0, 0.6], 0.3, 1.0)
        phonemes['en'] = morph_key([1.2, 0.9, 0.6, 0.4], 0.3, 1.0)
        phonemes['in'] = morph_key([0.3, 0.3, 0.4, 0.5], 0.4, 1.0)
        phonemes['un'] = morph_key([0.8, 1.0, 0.7, 0.5], 0.3, 1.0)
        phonemes['ün'] = morph_key([0.4, 0.5, 0.6, 0.7], 0.4, 1.0)
        phonemes['ang'] = morph_key([1.8, 1.4, 1.2, 0.8], 0.6, 1.0)
        phonemes['eng'] = morph_key([1.2, 1.0, 0.8, 0.6], 0.6, 1.0)
        phonemes['ing'] = morph_key([0.3, 0.4, 0.5, 0.6], 0.7, 1.0)
        phonemes['ong'] = morph_key([0.7, 1.2, 1.0, 0.7], 0.5, 1.0)
        phonemes['b'] = morph_key([0.0, 0.0, 0.5, 1.0], 0.0, 1.0)
        phonemes['d'] = morph_key([0.2, 0.0, 0.0, 0.8], 0.0, 1.0)
        phonemes['g'] = morph_key([0.8, 0.0, 0.0, 0.6], 0.0, 1.0)
        phonemes['p'] = morph_key([0.0, 0.0, 0.4, 0.9], 0.0, 0.8)
        phonemes['t'] = morph_key([0.2, 0.0, 0.0, 0.7], 0.0, 0.8)
        phonemes['k'] = morph_key([0.8, 0.0, 0.0, 0.5], 0.0, 0.8)
        phonemes['z'] = morph_key([0.1, 0.0, 0.3, 0.8], 0.0, 0.7)
        phonemes['zh'] = morph_key([0.2, 0.0, 0.2, 0.7], 0.0, 0.7)
        phonemes['j'] = morph_key([0.3, 0.0, 0.1, 0.6], 0.0, 0.7)
        phonemes['c'] = morph_key([0.1, 0.0, 0.2, 0.7], 0.0, 0.6)
        phonemes['ch'] = morph_key([0.2, 0.0, 0.1, 0.6], 0.0, 0.6)
        phonemes['q'] = morph_key([0.3, 0.0, 0.0, 0.5], 0.0, 0.6)
        phonemes['f'] = morph_key([0.0, 0.3, 0.5, 0.9], 0.0, 0.0)
        phonemes['s'] = morph_key([0.1, 0.2, 0.4, 0.8], 0.0, 0.0)
        phonemes['sh'] = morph_key([0.2, 0.2, 0.3, 0.7], 0.0, 0.0)
        phonemes['x'] = morph_key([0.3, 0.3, 0.2, 0.6], 0.0, 0.0)
        phonemes['h'] = morph_key([0.9, 0.7, 0.4, 0.2], 0.0, 0.0)
        phonemes['r'] = morph_key([0.4, 0.3, 0.2, 0.5], 0.0, 0.9)
        phonemes['m'] = morph_key([0.0, 0.0, 0.4, 0.8], 0.9, 1.0)
        phonemes['n'] = morph_key([0.2, 0.0, 0.2, 0.7], 0.9, 1.0)
        phonemes['ng'] = morph_key([0.8, 0.5, 0.3, 0.2], 0.8, 1.0)
        phonemes['l'] = morph_key([0.3, 0.6, 0.5, 0.7], 0.0, 1.0)

        return phonemes

    def callmorph(self, string):
        self.pm.loadpinyin()
        phonemes = self.pm.callstring(string)
        
        for i in phonemes:
            if i != "[hangup]":
                pass