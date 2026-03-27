import random
import winsound
import tempfile
import wave
import struct
import math
import os
from time import time

class synthesizer:
    def __init__(self):
        pass
    
    def play_func(self, func, duration):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            filename = tmp.name
        
        try:
            sample_rate = 8000
            with wave.open(filename, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                
                for i in range(int(duration * sample_rate)):
                    print(f"\rLoading {[".   ", "..  ", "... ", "...."][int(time()*4)%4]} {int(i / duration / sample_rate * 100)+1}%", end="      ")
                    t = i / sample_rate
                    value = func(t)
                    value = math.tanh(value)
                    sample = int(value * 32767)
                    wav.writeframes(struct.pack('<h', sample))
            
            winsound.PlaySound(filename, winsound.SND_FILENAME)
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def sin_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        if x == 0 or (index * 2 + 1 < len(reg) and reg[index * 2] != list(fft.keys())):
            frequencies = list(fft.keys())
            amplitudes = list(fft.values())
            angular_freqs = [2 * math.pi * f for f in frequencies]
            
            while len(reg) <= index * 2 + 1:
                reg.append(0.0)
            
            reg[index * 2] = angular_freqs
            reg[index * 2 + 1] = amplitudes
        
        if index * 2 + 1 < len(reg) and reg[index * 2] != 0.0:
            freqs = reg[index * 2]
            amps = reg[index * 2 + 1]
            
            result = 0.0
            for i in range(len(freqs)):
                result += math.sin(freqs[i] * x) * amps[i]
            return result
        
        return 0.0
    
    def bool_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        if x == 0 or (index * 2 + 1 < len(reg) and reg[index * 2] != list(fft.keys())):
            frequencies = list(fft.keys())
            amplitudes = list(fft.values())
            angular_freqs = [2 * math.pi * f for f in frequencies]
            
            while len(reg) <= index * 2 + 1:
                reg.append(0.0)
            
            reg[index * 2] = angular_freqs
            reg[index * 2 + 1] = amplitudes
        
        if index * 2 + 1 < len(reg) and reg[index * 2] != 0.0:
            freqs = reg[index * 2]
            amps = reg[index * 2 + 1]
            
            result = 0.0
            for i in range(len(freqs)):
                sign = 1.0 if math.sin(freqs[i] * x) >= 0 else -1.0
                result += sign * amps[i]
            return result
        
        return 0.0
    
    def triangle_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        if x == 0 or (index * 2 + 1 < len(reg) and reg[index * 2] != list(fft.keys())):
            frequencies = list(fft.keys())
            amplitudes = list(fft.values())
            angular_freqs = [2 * math.pi * f for f in frequencies]
            
            while len(reg) <= index * 2 + 1:
                reg.append(0.0)
            
            reg[index * 2] = angular_freqs
            reg[index * 2 + 1] = amplitudes
        
        if index * 2 + 1 < len(reg) and reg[index * 2] != 0.0:
            freqs = reg[index * 2]
            amps = reg[index * 2 + 1]
            
            result = 0.0
            for i in range(len(freqs)):
                result += (2 / math.pi) * math.asin(math.sin(freqs[i] * x)) * amps[i]
            return result
        
        return 0.0
    
    def random_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        if x == 0 or (index * 2 + 1 < len(reg) and reg[index * 2] != list(fft.keys())):
            frequencies = list(fft.keys())
            amplitudes = list(fft.values())
            angular_freqs = [2 * math.pi * f for f in frequencies]
            
            while len(reg) <= index * 2 + 1:
                reg.append(0.0)
            
            reg[index * 2] = angular_freqs
            reg[index * 2 + 1] = amplitudes
        
        if index * 2 + 1 < len(reg) and reg[index * 2] != 0.0:
            def f(t, a):
                ret = 0.0
                deep = 5
                for i in range(1, deep+1):
                    random.seed(int(t * 1000) + i * int(a * 100))
                    amp = 1.0 / (i ** 1.5)
                    phase = random.uniform(0, 2 * math.pi)
                    ret += random.uniform(0, amp) * math.sin(a*t+phase)
                return ret / deep
            
            freqs = reg[index * 2]
            amps = reg[index * 2 + 1]
            
            result = 0.0
            for i in range(len(freqs)):
                result += f(x, freqs[i]) * amps[i]
            return result
        
        return 0.0

    def mixed(self, ffts: list, weights: list, x):
        total_n = 0.0
        total_weight = 0.0
        
        for i, (fft, weight) in enumerate(zip(ffts, weights)):
            contribution = self.sin_fft(fft, x, i)
            total_n += weight * contribution
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        return total_n / total_weight

    
    def envelope_adsr(self, t, duration, attack=0.1, decay=0.1, sustain=0.7, release=0.2):
        if t < attack:
            return t / attack
        elif t < attack + decay:
            return 1.0 - (1.0 - sustain) * ((t - attack) / decay)
        elif t < duration - release:
            return sustain
        else:
            release_start = duration - release
            if t >= release_start:
                return sustain * (1.0 - (t - release_start) / release)
        return 0.0

    def envelope_linear(self, t, duration, start=0.0, end=1.0):
        return start + (end - start) * (t / duration)

    def envelope_exp(self, t, duration, start=1.0, end=0.0, decay=3.0):
        return start * math.exp(-decay * t / duration)

    def envelope_gate(self, t, duration, start=0.1, end=0.9):
        if start <= t <= end:
            return 1.0
        return 0.0

    def envelope_tremolo(self, t, freq=5.0, depth=0.5):
        return 1.0 - depth * 0.5 + depth * 0.5 * math.sin(2 * math.pi * freq * t)

    def envelope_multistage(self, t, points):
        if t <= points[0][0]:
            return points[0][1]
        
        for i in range(len(points) - 1):
            t1, v1 = points[i]
            t2, v2 = points[i + 1]
            if t1 <= t <= t2:
                ratio = (t - t1) / (t2 - t1)
                return v1 + (v2 - v1) * ratio
        
        return points[-1][1]

    
    def with_envelope(self, sound_func, envelope_func, *envelope_args):
        def wrapped(t):
            sound = sound_func(t)
            env = envelope_func(t, *envelope_args)
            return sound * env
        return wrapped

    def func_filter(self, x, fres):
        sigmoid = lambda x: 1 / (1 + math.e ** -x)
        return sigmoid(x - fres) * (1 - sigmoid(x - fres)) * 4 * x

    def time_weight(self, nowtime, totaltime):
        return max(0, (totaltime - nowtime) / totaltime)
    
    def random_weight(self, probability):
        return min(1, max(0, random.uniform(0.5-probability, 0.5+probability)))

    def exp_weight(self, nowtime, totaltime, decay=2.0):
        return math.exp(-decay * nowtime / totaltime)

    def sin_weight(self, nowtime, totaltime):
        return abs(math.sin(math.pi * nowtime / totaltime))
    
class parrot:
    def __init__(self, length: float = 20, n_segments: int = 20, nasal_length_ratio: float = 0.5):
        self.n = n_segments
        self.total_length = length / 100
        self.dx = self.total_length / n_segments
        self.areas = [2.0] * n_segments
        self.c = 340
        self.nasal_length_ratio = nasal_length_ratio
        self.nasal_length = self.total_length * nasal_length_ratio
        self.nasal_area = 1.0
        self.nasal_coupling = 0.3
        self.voiced = 1.0
        self.s = synthesizer()
        self.key = {}
        self.obstruction_threshold = 0.0
        self.noise_threshold = 0.25
        
    def check_obstructions(self):
        min_area = min(self.areas)
        min_index = self.areas.index(min_area)
        if min_area < self.obstruction_threshold:
            return min_index
        return None
    
    def is_obstruction_before_velum(self, obstruction_pos):
        velum_position = int(self.n * 0.66)
        return obstruction_pos < velum_position
    
    def gate(self, alpha, fft: dict[float, float], index: int = 0):
        return lambda t: max(min(1, alpha)*self.s.triangle_fft(fft, t, index*2), (1-min(1, alpha))*self.s.random_fft(fft, t, index*2+1))

    def tube(self, len):
        depth = 5
        resonances = {}
        for i in range(1, depth + 1):
            freq = (2 * i - 1) * self.c / (len * 4)
            amplitude = 1 / i
            resonances[freq] = amplitude
        return resonances
    
    def speak(self, pitch: float = 0.0):
        fft_dict = {}
        obstruction_pos = self.check_obstructions()
        avg_area = sum(self.areas) / len(self.areas)
        
        noise_intensity = 0.0
        for area in self.areas:
            if area < self.noise_threshold:
                noise_intensity += (self.noise_threshold - area) / self.noise_threshold
        noise_intensity = min(1.0, noise_intensity / len(self.areas) * 2)
        
        if obstruction_pos is not None:
            if self.nasal_area >= 0.0 and self.is_obstruction_before_velum(obstruction_pos):
                nasal_resonances = self.tube(self.nasal_length)
                for freq, amp in nasal_resonances.items():
                    fft_dict[freq * 2**(pitch/2)] = amp * self.nasal_coupling * obstruction_pos / 4
                if noise_intensity > 0:
                    for _ in range(3):
                        noise_freq = random.uniform(2000, 5000)
                        fft_dict[noise_freq] = noise_intensity * 0.2 * obstruction_pos / 4
            else:
                if noise_intensity > 0.5:
                    for _ in range(2):
                        noise_freq = random.uniform(1000, 3000)
                        fft_dict[noise_freq] = noise_intensity * 0.1
        else:
            base_freq = 340 / (2 * self.total_length) * random.uniform(0.5, 1.5)
            tube_resonances = self.tube(self.total_length)
            
            for freq, amp in tube_resonances.items():
                adjusted_amp = amp * (avg_area / 2.0)
                fft_dict[freq * 2**(pitch/2)] = adjusted_amp
            
            if self.nasal_area >= 0.0:
                nasal_resonances = self.tube(self.nasal_length)
                for freq, amp in nasal_resonances.items():
                    if freq in fft_dict:
                        fft_dict[freq * 2**(pitch/2)] += amp * self.nasal_coupling * 0.5
                    else:
                        fft_dict[freq * 2**(pitch/2)] = amp * self.nasal_coupling * 0.5
            
            for harmonic in range(1, 6):
                freq = base_freq * harmonic
                if freq > 8000:
                    break
                amplitude = 1.0 / (harmonic ** 1.5)
                segment_variation = 1.0
                for i, area in enumerate(self.areas):
                    if 100 * (i + 1) < freq < 200 * (i + 2):
                        segment_variation *= (area / 2.0)
                amplitude *= segment_variation
                for existing_freq in fft_dict:
                    if abs(freq - existing_freq) < 50:
                        amplitude += fft_dict[existing_freq] * 0.3
                        break
                if freq in fft_dict:
                    fft_dict[freq * 2**(pitch/2)] = max(fft_dict[freq * 2**(pitch/2)], amplitude)
                else:
                    fft_dict[freq * 2**(pitch/2)] = amplitude
            
            if noise_intensity > 0:
                for _ in range(int(5 * noise_intensity)):
                    noise_freq = random.uniform(2000, 6000)
                    noise_amp = random.uniform(0.05, 0.15) * (avg_area / 3.0) * noise_intensity
                    fft_dict[noise_freq] = noise_amp
        
        return (self.voiced, fft_dict)
    
    def add_key(self, name, val):
        self.key[name] = val
        return self
    
    def get_key(self, key):
        self.areas = list(key)[0]
        self.nasal_area = list(key)[1]
        self.voiced = list(key)[2]
        return self

    def get_key_for_(self, name):
        self.get_key(self.key[name])
        return self
    
    def syllable(self, vowel, consonant, t, duration):
        v = list(self.key[vowel])
        c = list(self.key[consonant])

        ratio = t/duration

        return morph_key([ratio*i + (1 - ratio)*j for i,j in zip(v[0], c[0])], ratio*v[1] + (1-ratio)*c[1], ratio*v[2] + (1-ratio)*c[2])
    
class morph_key:
    def __init__(self, areas, nasal, voiced):
        self.areas = areas
        self.nasal = nasal
        self.voiced = voiced
    
    def __list__(self):
        return [self.areas, self.nasal, self.voiced]
    
    def __iter__(self):
        return iter([self.areas, self.nasal, self.voiced])

if __name__ == "__main__":
    p = parrot(12, 4)

    T = 4

    
    p.add_key("b", morph_key([0.1,0.1,0.5,0.0], 0.0, 1.0))
    p.add_key("a", morph_key([1.0,1.0,1.5,0.75], 0.0, 1.0))

    f = lambda t: p.gate(*p.get_key(p.syllable("b","a", (min((t*2)%4, 2)), 2)).speak())(t) * p.s.envelope_adsr(t%2, 2)

    p.s.play_func(f, T)