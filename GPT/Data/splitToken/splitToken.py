from pathlib import Path
import math
import glob
import json
import pickle
from collections import defaultdict
from datetime import datetime

class ChineseTokenizer:
    def __init__(self, max_token_len=6, min_freq_factor=50):
        self.max_token_len = max_token_len
        self.min_freq_factor = min_freq_factor
        self.tokens = set()
        self.ngram_freq = {}
        self.model_info = {}
        
    def get_models_dir(self):
        current_dir = Path(__file__).parent
        models_dir = current_dir / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir
    
    def load_texts(self, folder_path):
        txt_files = glob.glob(str(folder_path / "*.txt"))
        texts = []
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
                        print(f"Loaded: {Path(file_path).name} ({len(content)} chars)")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        return ''.join(texts)
    
    def count_ngrams(self, text):
        ngram_freq = defaultdict(int)
        total_chars = len(text)
        
        print("Counting n-grams...")
        
        for n in range(1, self.max_token_len + 1):
            count = 0
            for i in range(total_chars - n + 1):
                substr = text[i:i+n]
                if substr.strip() and len(substr) == n:
                    ngram_freq[substr] += 1
                    count += 1
            print(f"  {n}-grams: {count}")
        
        return dict(ngram_freq), total_chars
    
    def calculate_cohesion(self, ngram, ngram_freq, total_chars):
        if len(ngram) < 2:
            return 0
        
        word_freq = ngram_freq.get(ngram, 0)
        if word_freq == 0:
            return 0
        
        max_ratio = 0
        for i in range(1, len(ngram)):
            left = ngram[:i]
            right = ngram[i:]
            
            left_freq = ngram_freq.get(left, 0)
            right_freq = ngram_freq.get(right, 0)
            
            if left_freq > 0 and right_freq > 0:
                expected_freq = (left_freq * right_freq) / total_chars
                if expected_freq > 0:
                    ratio = word_freq / expected_freq
                    max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def calculate_boundary_entropy(self, ngram, text):
        if len(ngram) < 2:
            return 0, 0
        
        left_chars = defaultdict(int)
        right_chars = defaultdict(int)
        pattern_len = len(ngram)
        text_len = len(text)
        
        for i in range(text_len - pattern_len + 1):
            if text[i:i+pattern_len] == ngram:
                if i > 0:
                    left_chars[text[i-1]] += 1
                if i + pattern_len < text_len:
                    right_chars[text[i+pattern_len]] += 1
        
        def calculate_entropy(char_dict):
            total = sum(char_dict.values())
            if total == 0:
                return 0
            entropy = 0
            for count in char_dict.values():
                prob = count / total
                entropy -= prob * math.log(prob)
            return entropy
        
        left_entropy = calculate_entropy(left_chars)
        right_entropy = calculate_entropy(right_chars)
        
        return left_entropy, right_entropy
    
    def extract_tokens(self, ngram_freq, total_chars, text):
        tokens = set()
        
        min_freq = max(3, total_chars // (self.min_freq_factor * 100))
        print(f"Minimum frequency threshold: {min_freq}")
        
        candidate_scores = {}
        
        print("Evaluating n-grams...")
        evaluated = 0
        
        for ngram, freq in ngram_freq.items():
            if len(ngram) == 1:
                tokens.add(ngram)
                continue
                
            if freq < min_freq:
                continue
                
            evaluated += 1
            if evaluated % 10000 == 0:
                print(f"  Evaluated {evaluated} n-grams...")
            
            cohesion = self.calculate_cohesion(ngram, ngram_freq, total_chars)
            if cohesion < 20:
                continue
            
            left_entropy, right_entropy = self.calculate_boundary_entropy(ngram, text)
            min_entropy = min(left_entropy, right_entropy)
            
            if min_entropy < 1.0:
                continue
            
            score = cohesion * min_entropy * math.log(freq + 1)
            candidate_scores[ngram] = score
        
        print(f"Found {len(candidate_scores)} candidates")
        
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Selecting tokens...")
        selected_count = 0
        
        for ngram, score in sorted_candidates:
            if len(ngram) > 1:
                is_substring = False
                for token in tokens:
                    if ngram in token and ngram != token:
                        is_substring = True
                        break
                
                if not is_substring:
                    tokens.add(ngram)
                    selected_count += 1
                    
                    if selected_count % 100 == 0:
                        print(f"  Selected {selected_count} tokens...")
        
        print(f"Selected {len(tokens)} tokens total")
        return tokens
    
    def save_model(self, model_name=None):
        models_dir = self.get_models_dir()
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"chinese_tokenizer_{timestamp}"
        
        model_path = models_dir / f"{model_name}.pkl"
        
        model_data = {
            'tokens': list(self.tokens),
            'ngram_freq': self.ngram_freq,
            'max_token_len': self.max_token_len,
            'min_freq_factor': self.min_freq_factor,
            'model_info': self.model_info
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        json_path = models_dir / f"{model_name}_info.json"
        json_data = {
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'num_tokens': len(self.tokens),
            'max_token_len': self.max_token_len,
            'min_freq_factor': self.min_freq_factor,
            'model_info': self.model_info
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to: {model_path}")
        return model_name
    
    def load_model(self, model_name):
        models_dir = self.get_models_dir()
        model_path = models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            available_models = [f.stem for f in models_dir.glob("*.pkl")]
            raise FileNotFoundError(f"Model '{model_name}' not found. Available: {available_models}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tokens = set(model_data['tokens'])
        self.ngram_freq = model_data['ngram_freq']
        self.max_token_len = model_data['max_token_len']
        self.min_freq_factor = model_data['min_freq_factor']
        self.model_info = model_data.get('model_info', {})
        
        print(f"Model '{model_name}' loaded")
        print(f"Vocabulary: {len(self.tokens)} tokens")
        return True
    
    def list_models(self):
        models_dir = self.get_models_dir()
        models = []
        
        for pkl_file in models_dir.glob("*.pkl"):
            json_file = models_dir / f"{pkl_file.stem}_info.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                models.append(info)
        
        return models
    
    def train(self, folder_path, save_model=True, model_name=None):
        print("Training Chinese tokenizer...")
        text = self.load_texts(folder_path)
        
        if not text:
            print("Error: No text data")
            return set()
        
        print(f"Total text: {len(text)} characters")
        
        self.ngram_freq, total_chars = self.count_ngrams(text)
        print(f"Unique n-grams: {len(self.ngram_freq)}")
        
        self.tokens = self.extract_tokens(self.ngram_freq, total_chars, text)
        print(f"Final tokens: {len(self.tokens)}")
        
        self.model_info = {
            'training_date': datetime.now().isoformat(),
            'text_length': total_chars,
            'unique_ngrams': len(self.ngram_freq),
            'num_tokens': len(self.tokens)
        }
        
        if save_model:
            saved_name = self.save_model(model_name)
            self.model_info['saved_as'] = saved_name
        
        return self.tokens
    
    def tokenize(self, text, method='max_match'):
        if not self.tokens:
            raise ValueError("Please train or load model")
        
        if method == 'max_match':
            return self._max_match_tokenize(text)
        elif method == 'forward':
            return self._forward_max_match(text)
        else:
            return self._simple_tokenize(text)
    
    def _max_match_tokenize(self, text):
        tokens = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            matched = False
            for length in range(min(self.max_token_len, text_len - i), 0, -1):
                word = text[i:i+length]
                if word in self.tokens:
                    tokens.append(word)
                    i += length
                    matched = True
                    break
            
            if not matched:
                tokens.append(text[i])
                i += 1
        
        return tokens
    
    def _forward_max_match(self, text):
        tokens = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            max_word = text[i]
            max_len = 1
            
            for length in range(2, min(self.max_token_len, text_len - i) + 1):
                word = text[i:i+length]
                if word in self.tokens:
                    max_word = word
                    max_len = length
            
            tokens.append(max_word)
            i += max_len
        
        return tokens
    
    def _simple_tokenize(self, text):
        tokens = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            found = False
            for length in range(min(self.max_token_len, text_len - i), 0, -1):
                word = text[i:i+length]
                if word in self.tokens:
                    tokens.append(word)
                    i += length
                    found = True
                    break
            
            if not found:
                tokens.append(text[i])
                i += 1
        
        return tokens


if __name__ == "__main__":
    """tokenizer = ChineseTokenizer(max_token_len=6, min_freq_factor=50)
    
    filespath = Path(__file__).parent.parent / "Text"
    
    if filespath.exists() and any(filespath.glob("*.txt")):
        tokens = tokenizer.train(filespath, save_model=True)
        
        print("\nTop 100 tokens by frequency:")
        sorted_tokens = sorted(
            [(t, tokenizer.ngram_freq.get(t, 0)) for t in tokens if len(t) > 1],
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (token, freq) in enumerate(sorted_tokens[:100]):
            print(f"{i+1:3d}. '{token}' (freq: {freq})")
        
        print(f"\nTotal tokens: {len(tokens)}")
        print(f"Single chars: {len([t for t in tokens if len(t) == 1])}")
        print(f"Multi-chars: {len([t for t in tokens if len(t) > 1])}")
        
        test_texts = [
            "这是测试文本",
            "三体世界",
            "黑暗森林",
            "物理学不存在了",
            "叶文洁看着红岸基地"
        ]
        
        for test_text in test_texts:
            segmented = tokenizer.tokenize(test_text, method='max_match')
            print(f"\nTest: '{test_text}'")
            print(f"  Result: {segmented}")
    else:
        print(f"Text directory not found: {filespath}")
        print("Loading existing models...")
        
        models = tokenizer.list_models()
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model['model_name']} ({model['num_tokens']} tokens)")
            
            latest_model = sorted(models, key=lambda x: x['created_at'], reverse=True)[0]
            tokenizer.load_model(latest_model['model_name'])
            
            test_texts = [
                "这是测试文本",
                "三体世界",
                "黑暗森林"
            ]
            
            for test_text in test_texts:
                segmented = tokenizer.tokenize(test_text)
                print(f"\nTest: '{test_text}' -> {segmented}")
        else:
            print("No models found")"""
    
    text = """从去年起，仿佛听得有人说我是仇猫的。那根据自然是在我的那一篇《兔和猫》；这是自画招供，当然无话可说，——但倒也毫不介意。一到今年，我可很有点担心了。我是常不免于弄弄笔墨的，写了下来，印了出去，对于有些人似乎总是搔着癢处的时候少，碰着痛处的时候多。万一不谨，甚而至于得罪了名人或名教授，或者更甚而至于得罪了“负有指导青年责任的前辈”之流，可就危险已极。为什么呢？因为这些大脚色是“不好惹”的。怎地“不好惹”呢？就是怕要浑身发热之后，做一封信登在报纸上，广告道：“看哪！狗不是仇猫的么？鲁迅先生却自己承认是仇猫的，而他还说要打‘落水狗’！”①这“逻辑”的奥义，即在用我的话，来证明我倒是狗，于是而凡有言说，全都根本推翻，即使我说二二得四，三三见九，也没有一字不错。这些既然都错，则绅士口头的二二得七，三三见千等等，自然就不错了。
我于是就间或留心着查考它们成仇的“动机”。
"""

    model = ChineseTokenizer()
    model.load_model(str("model"))
    ret = model.tokenize(text)
    print(ret)