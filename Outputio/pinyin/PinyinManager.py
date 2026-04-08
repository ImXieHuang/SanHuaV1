import json

class PinyinManager:
    def __init__(self):
        self.INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                        'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        self.pinyin = []


    def extract_initial_final(self, pinyin):
        tone_marks = {'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
                    'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
                    'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
                    'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
                    'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
                    'ǖ': 'ü', 'ǘ': 'ü', 'ǚ': 'ü', 'ǜ': 'ü',
                    'ü': 'ü', 'ń': 'n', 'ň': 'n', 'ǹ': 'n', 'ḿ': 'm'}
        
        pure = ''.join(tone_marks.get(c, c) for c in pinyin)
        
        if not pure:
            return ('', '')
        
        if len(pure) == 1 or pure[0] not in 'bcdfghjklmnpqrstwxyz':
            return ('', pure)
        
        initial = ''
        for i in self.INITIALS:
            if pure.startswith(i):
                initial = i
                break
        
        final = pure[len(initial):] if initial else pure
        return (initial, final)


    def parse_pinyin_line(self, line):
        line = line.strip()
        if not line or line.startswith('#'):
            return None
        
        if '#' in line:
            line = line.split('#')[0].strip()
        
        if not line or not line.startswith('U+'):
            return None
        
        parts = line.split(':', 1)
        if len(parts) != 2:
            return None
        
        code_point = parts[0].strip()
        pinyin_part = parts[1].strip()
        
        try:
            char = chr(int(code_point[2:], 16))
        except (ValueError, OverflowError):
            return None
        
        pinyins = [p.strip() for p in pinyin_part.split(',') if p.strip()]
        
        return (char, pinyins)

    def process(self):
        input_file = "\\".join(__file__.split("\\")[:-1])+'\\pinyin.txt'
        output_file = "\\".join(__file__.split("\\")[:-1])+'\\pinyin.json'
        
        result = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self.parse_pinyin_line(line)
                if not parsed:
                    continue
                
                char, pinyins = parsed
                char_result = {
                    'char': char,
                    'readings': []
                }
                
                for pinyin in pinyins:
                    initial, final = self.extract_initial_final(pinyin)
                    char_result['readings'].append({
                        'pinyin': pinyin,
                        'initial': initial,
                        'final': final
                    })
                
                result.append(char_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f" - Processing completed! Total {len(result)} Chinese characters, saved to {output_file}")

    def loadpinyin(self):
        file = "\\".join(__file__.split("\\")[:-1])+'\\pinyin.json'
        with open(file, 'r', encoding='utf-8') as f:
            self.pinyin = json.load(f)
        return self.pinyin

    def callpinyin(self, char):
        for i in self.pinyin:
            if i["char"] == char:
                break
        else:
            return "[hangup]"
        return (i["readings"][0]["initial"], i["readings"][0]["final"])

    def callstring(self, string):
        readings = []
        for i in string:
            for j in self.pinyin:
                if j["char"] == i:
                    break
            else:
                if len(readings) >= 1 and readings[-1] != "[hangup]":
                    readings.append("[hangup]")
                continue
            readings.append((j["readings"][0]["initial"], j["readings"][0]["final"]))
        
        return readings

if __name__ == '__main__':
    pm = PinyinManager()
    pm.process()

    pm.loadpinyin()
    print(pm.callstring('你好！...你叫什么名字？'))