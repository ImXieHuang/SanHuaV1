import random
import ctypes
import tempfile
import wave
import struct
import math
import os
import sys
from time import time

winmm = ctypes.windll.winmm
SND_FILENAME = 0x00020000
SND_SYNC = 0x0000

class synthesizer:
    def __init__(self):
        self._wave_table_cache = {}
        self._sample_rate = 6500
        
    def _next_power_of_2(self, n):
        p = 1
        while p < n:
            p <<= 1
        return p
    
    def _fft(self, a, invert):
        n = len(a)
        if n == 1:
            return a
        
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        
        length = 2
        while length <= n:
            ang = 2 * math.pi / length * (-1 if invert else 1)
            wlen = complex(math.cos(ang), math.sin(ang))
            for i in range(0, n, length):
                w = 1+0j
                for j in range(i, i + length // 2):
                    u = a[j]
                    v = a[j + length // 2] * w
                    a[j] = u + v
                    a[j + length // 2] = u - v
                    w *= wlen
            length <<= 1
        
        if invert:
            for i in range(n):
                a[i] /= n
        return a
    
    def _build_wave_table(self, fft_dict, duration, wave_type):
        n_samples = int(duration * self._sample_rate)
        df = self._sample_rate / n_samples
        n_fft = self._next_power_of_2(n_samples)
        spec_len = n_fft // 2 + 1
        
        spec = [0j] * spec_len
        for freq, amp in fft_dict.items():
            if amp == 0:
                continue
            bin_idx = int(round(freq / df))
            if 0 <= bin_idx < spec_len:
                spec[bin_idx] = complex(amp, 0)
        
        full_spec = [0j] * n_fft
        for i in range(spec_len):
            full_spec[i] = spec[i]
        for i in range(1, spec_len - 1):
            full_spec[n_fft - i] = spec[i].conjugate()
        
        time_domain = self._fft(full_spec, invert=True)
        base_wave = [time_domain[i].real for i in range(n_samples)]
        
        if wave_type == 'sin':
            wave_table = base_wave
        elif wave_type == 'bool':
            wave_table = [1.0 if v >= 0 else -1.0 for v in base_wave]
        elif wave_type == 'triangle':
            phase = [0.0] * n_samples
            for i in range(1, n_samples):
                phase[i] = phase[i-1] + base_wave[i] * 0.01
            wave_table = [(2 / math.pi) * math.asin(math.sin(p)) for p in phase]
        elif wave_type == 'random':
            wave_table = base_wave
            random.seed(0)
            for i in range(n_samples):
                wave_table[i] = wave_table[i] * random.uniform(0.5, 1.5)
        else:
            wave_table = base_wave
        
        if wave_table:
            max_val = max(abs(max(wave_table)), abs(min(wave_table)))
            if max_val > 1e-6:
                gain = 0.9 / max_val
                wave_table = [v * gain for v in wave_table]
        
        return wave_table
    
    def sin_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        cache_key = (id(fft), index, 'sin')
        
        if cache_key not in self._wave_table_cache:
            duration = 1.0
            wave_table = self._build_wave_table(fft, duration, 'sin')
            self._wave_table_cache[cache_key] = wave_table
        
        wave_table = self._wave_table_cache[cache_key]
        idx = int(x * self._sample_rate) % len(wave_table)
        return wave_table[idx]
    
    def bool_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        cache_key = (id(fft), index, 'bool')
        
        if cache_key not in self._wave_table_cache:
            duration = 1.0
            wave_table = self._build_wave_table(fft, duration, 'bool')
            self._wave_table_cache[cache_key] = wave_table
        
        wave_table = self._wave_table_cache[cache_key]
        idx = int(x * self._sample_rate) % len(wave_table)
        return wave_table[idx]
    
    def triangle_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        cache_key = (id(fft), index, 'triangle')
        
        if cache_key not in self._wave_table_cache:
            duration = 1.0
            wave_table = self._build_wave_table(fft, duration, 'triangle')
            self._wave_table_cache[cache_key] = wave_table
        
        wave_table = self._wave_table_cache[cache_key]
        idx = int(x * self._sample_rate) % len(wave_table)
        return wave_table[idx]

    def random_fft(self, fft: dict[float, float], x, index: int = 0, reg = []):
        cache_key = (id(fft), index, 'random')
        
        if cache_key not in self._wave_table_cache:
            duration = 1.0
            wave_table = self._build_wave_table(fft, duration, 'random')
            self._wave_table_cache[cache_key] = wave_table
        
        wave_table = self._wave_table_cache[cache_key]
        idx = int(x * self._sample_rate) % len(wave_table)
        return wave_table[idx]

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
    
    def play_func(self, func, duration):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            filename = tmp.name
        
        try:
            sample_rate = 12000
            total_samples = int(duration * sample_rate)
            
            samples = [0.0] * total_samples
            
            for i in range(total_samples):
                print(f"\rLoading {['.   ', '..  ', '... ', '....'][int(time()*4)%4]} {int(i / total_samples * 100)+1}%", end="      ")
                sys.stdout.flush()
                t = i / sample_rate
                samples[i] = func(t)
            
            for i in range(total_samples):
                samples[i] = math.tanh(samples[i])
            
            with wave.open(filename, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                
                frames = bytearray()
                for sample in samples:
                    frames.extend(struct.pack('<h', int(sample * 32767)))
                
                wav.writeframes(frames)
            
            winmm.PlaySoundW(filename, None, SND_FILENAME | SND_SYNC)
            
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


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
        self.sample_rate = 6500
        self.segment_delay = max(1, int((self.total_length / self.n / self.c) * self.sample_rate))
        self.source_buffer = []
        
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
    
    def kl_synthesis(self, source, duration):
        n_segments = self.n
        segment_length = self.total_length / n_segments
        delay_samples = max(1, int(segment_length / self.c * self.sample_rate))
        
        k = []
        for i in range(n_segments - 1):
            a1 = self.areas[i]
            a2 = self.areas[i + 1]
            if a1 + a2 == 0:
                k.append(0.0)
            else:
                k.append((a1 - a2) / (a1 + a2))
        
        forward = [0.0] * (n_segments + 1)
        backward = [0.0] * (n_segments + 1)
        delay_line_f = []
        delay_line_b = []
        for i in range(n_segments):
            delay_line_f.append([0.0] * delay_samples)
            delay_line_b.append([0.0] * delay_samples)
            for j in range(delay_samples):
                delay_line_f[i][j] = 0.0
                delay_line_b[i][j] = 0.0
        delay_idx = 0
        
        total_samples = int(duration * self.sample_rate)
        output = [0.0] * total_samples
        
        for n in range(total_samples):
            src = source[n] if n < len(source) else 0.0
            
            u_plus = [0.0] * (n_segments + 1)
            u_minus = [0.0] * (n_segments + 1)
            
            u_plus[0] = src + backward[0]
            
            for i in range(n_segments - 1):
                u_plus[i + 1] = u_plus[i] * (1 + k[i]) - backward[i + 1] * k[i]
            
            u_minus[n_segments] = 0.0
            
            for i in range(n_segments - 1, 0, -1):
                u_minus[i] = u_minus[i + 1] * (1 - k[i - 1]) + u_plus[i - 1] * k[i - 1]
            
            for i in range(n_segments):
                delay_line_f[i][delay_idx] = u_plus[i]
                delay_line_b[i][delay_idx] = u_minus[i + 1]
            
            for i in range(n_segments):
                forward[i] = delay_line_f[i][(delay_idx - 1) % delay_samples]
                backward[i] = delay_line_b[i][(delay_idx - 1) % delay_samples]
            
            output[n] = forward[n_segments - 1]
            delay_idx = (delay_idx + 1) % delay_samples
        
        return output
    
    def generate_source(self, duration, f0=120):
        total_samples = int(duration * self.sample_rate)
        source = [0.0] * total_samples
        period = int(self.sample_rate / f0) if f0 > 0 else 100
        if period < 1:
            period = 1
        
        voiced = self.voiced > 0.5
        if voiced:
            for n in range(total_samples):
                if n % period == 0:
                    source[n] = 1.0
                else:
                    source[n] = 0.0
        else:
            for n in range(total_samples):
                source[n] = random.uniform(-0.3, 0.3)
        
        return source
    
    def speak(self, pitch: float = 0.0):
        f0 = 120 * (2 ** (pitch / 12))
        duration = 1.0
        source = self.generate_source(duration, f0)
        output = self.kl_synthesis(source, duration)
        
        fft_dict = {}
        n_samples = len(output)
        if n_samples > 0:
            window_size = min(512, n_samples)
            for i in range(min(20, window_size // 2)):
                freq = i * self.sample_rate / window_size
                if freq > 50 and freq < 5000:
                    magnitude = 0.0
                    for j in range(window_size):
                        if j < n_samples:
                            magnitude += abs(output[j]) * math.sin(2 * math.pi * freq * j / self.sample_rate)
                    if magnitude > 0.01:
                        fft_dict[freq] = magnitude * 10
        
        if len(fft_dict) == 0:
            fft_dict[440] = 0.5
        
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