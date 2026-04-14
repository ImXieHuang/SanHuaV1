try:
    from . import CCAT
except:
    import CCAT
import random
from typing import Union, List
import sys
from pathlib import Path
from math import sin, cos
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)
import Vector as vector
from Vector.vector import Vector
import mathexpand as mexp

def NewCCATransformer(Texts: Union[List[str], str], dim: int=8) -> CCAT.CCATransformer:
    if not isinstance(Texts, list):
        return {Texts: {Vector([random.uniform(-1, 1) for _ in range(dim)]): Vector([random.uniform(-1, 1) for _ in range(dim)])}}
    database = {}
    for i in Texts:
        database[i] = {Vector([random.uniform(-1, 1) for _ in range(dim)]): Vector([random.uniform(-1, 1) for _ in range(dim)])}
    return CCAT.CCATransformer(database, CCAT.Const.E, dim)

def FusionCCATransformer(CAT1: CCAT.CCATransformer, CAT2: CCAT.CCATransformer) -> CCAT.CCATransformer:
    FusionDatabsase = {}
    Keys = list(set(CAT1.get_tokens()) | set(CAT2.get_tokens()))
    for i in Keys:
        FusionDatabsase[i] = {}
        if i in CAT1.get_tokens() and i in CAT2.get_tokens():
            Query = list(set(CAT1.database[i]) | set(CAT2.database[i]))
            for j in Query:
                if j in CAT1.database[i] and j in CAT2.database[i]:
                    FusionDatabsase[i][j] = mexp.div(mexp.add(CAT1.database[i][j], CAT2.database[i][j]), 2)
                elif j in CAT1.database[i]:
                    FusionDatabsase[i][j] = CAT1.database[i][j]
                else:
                    FusionDatabsase[i][j] = CAT2.database[i][j]
        elif i in CAT1.get_tokens():
            FusionDatabsase[i] = CAT1.database[i]
        else:
            FusionDatabsase[i] = CAT2.database[i]
    return CCAT.CCATransformer(FusionDatabsase, CAT1.temperature / 2 + CAT2.temperature / 2)

def get_meaning_of_tokens_for_(ccat: CCAT.CCATransformer, tokens: List[str]) -> List[Vector]:
    influence_graph = [[0.0 for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            influence_graph[i][j] = vector.dot(ccat.get_query_for_(tokens[i]), ccat.get_key_for_(tokens[j]))
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if j >= i:
                influence_graph[i][j] = -float('inf')
    for i in range(len(tokens)):
        row = influence_graph[i]
        max_val = max(row)
        exp_row = [CCAT.Const.E ** (val - max_val) for val in row]
        sum_exp = sum(exp_row)
        influence_graph[i] = [val / sum_exp for val in exp_row]
    bigQ_graph = [[Vector([0.0] * ccat.dim) for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for j in range(i + 1):
            if i == 0 and j == 0:
                bigQ_graph[i][j] = ccat.get_key_for_(tokens[i])
            else:
                vector_list = []
                for k in range(j):
                    components = []
                    for l in range(ccat.dim):
                        influence_val = influence_graph[i][k]
                        if isinstance(influence_val, float):
                            influence_val = influence_val
                        component_val = bigQ_graph[i][k].components[l]
                        if isinstance(component_val, float):
                            component_val = component_val
                        components.append(component_val * influence_val)
                    vector_list.append(Vector(components))
                if not vector_list:
                    big_Q = Vector([0.0] * ccat.dim)
                else:
                    big_Q = mexp.iterate(mexp.add, vector_list)
                if not isinstance(big_Q, Vector):
                    if isinstance(big_Q, (int, float)):
                        big_Q = Vector([0.0] * ccat.dim)
                    else:
                        try:
                            big_Q = Vector([float(big_Q)] * ccat.dim)
                        except:
                            big_Q = Vector([0.0] * ccat.dim)
                bigQ_graph[i][j] = ccat.get_value_for_(tokens[j], big_Q)
    meaning_vectors = []
    for i in range(len(tokens)):
        vector_list = []
        for j in range(len(tokens)):
            influence_val = influence_graph[i][j]
            if isinstance(influence_val, float):
                influence_val = influence_val
            if isinstance(bigQ_graph[i][j], Vector):
                vector_list.append(mexp.mul(bigQ_graph[i][j], influence_val))
        if not vector_list:
            meaning_vector = Vector([0.0] * ccat.dim)
        else:
            meaning_vector = mexp.iterate(mexp.add, vector_list)
        if not isinstance(meaning_vector, Vector):
            if isinstance(meaning_vector, (int, float)):
                meaning_vector = Vector([0.0] * ccat.dim)
            else:
                try:
                    meaning_vector = Vector([float(meaning_vector)] * ccat.dim)
                except:
                    meaning_vector = Vector([0.0] * ccat.dim)
        meaning_vectors.append(meaning_vector + Vector([sin(i/10000**(i/ccat.dim)) if i % 2 == 0 else cos(i/10000**(i/ccat.dim)) for i in range(ccat.dim)]))
    meaning_vectors[0] = ccat.get_key_for_(tokens[0])
    return meaning_vectors

def get_meaning_of_tokens_at_(ccat: CCAT.CCATransformer, AtQ: Vector, tokens: List[str]) -> List[Vector]:
    influence_graph = [[0.0 for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            influence_graph[i][j] = mexp.mul(vector.dot(ccat.get_query_for_(tokens[i]), ccat.get_key_for_(tokens[j])), vector.dot(ccat.get_query_for_(tokens[i]), AtQ))
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if j >= i:
                influence_graph[i][j] = -float('inf')
    for i in range(len(tokens)):
        row = influence_graph[i]
        max_val = max(row)
        exp_row = [CCAT.Const.E ** (val - max_val) for val in row]
        sum_exp = sum(exp_row)
        influence_graph[i] = [val / sum_exp for val in exp_row]
    bigQ_graph = [[Vector([0.0] * ccat.dim) for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for j in range(i + 1):
            if j == 0:
                bigQ_graph[i][j] = ccat.get_key_for_(tokens[i])
            else:
                vector_list = []
                for k in range(j):
                    components = []
                    for l in range(ccat.dim):
                        influence_val = influence_graph[i][k]
                        if isinstance(influence_val, float):
                            influence_val = influence_val
                        component_val = bigQ_graph[i][k].components[l]
                        if isinstance(component_val, float):
                            component_val = component_val
                        components.append(component_val * influence_val)
                    vector_list.append(Vector(components))
                if not vector_list:
                    big_Q = Vector([0.0] * ccat.dim)
                else:
                    big_Q = mexp.iterate(mexp.add, vector_list)
                if not isinstance(big_Q, Vector):
                    if isinstance(big_Q, (int, float)):
                        big_Q = Vector([0.0] * ccat.dim)
                    else:
                        try:
                            big_Q = Vector([float(big_Q)] * ccat.dim)
                        except:
                            big_Q = Vector([0.0] * ccat.dim)
                bigQ_graph[i][j] = ccat.get_value_for_(tokens[j], big_Q)
    meaning_vectors = []
    for i in range(len(tokens)):
        vector_list = []
        for j in range(len(tokens)):
            influence_val = influence_graph[i][j]
            if isinstance(influence_val, float):
                influence_val = influence_val
            if isinstance(bigQ_graph[i][j], Vector):
                vector_list.append(mexp.mul(bigQ_graph[i][j], influence_val))
        if not vector_list:
            meaning_vector = Vector([0.0] * ccat.dim)
        else:
            meaning_vector = mexp.iterate(mexp.add, vector_list)
        if not isinstance(meaning_vector, Vector):
            if isinstance(meaning_vector, (int, float)):
                meaning_vector = Vector([0.0] * ccat.dim)
            else:
                try:
                    meaning_vector = Vector([float(meaning_vector)] * ccat.dim)
                except:
                    meaning_vector = Vector([0.0] * ccat.dim)
        meaning_vectors.append(meaning_vector + Vector([sin(i/10000**(i/ccat.dim)) if i % 2 == 0 else cos(i/10000**(i/ccat.dim)) for i in range(ccat.dim)]))
    meaning_vectors[0] = ccat.get_key_for_(tokens[0])
    return meaning_vectors

def get_meaning_of_sentence_for_(ccat: CCAT.CCATransformer, tokens: List[str]) -> Vector:
    tokens_meaning = get_meaning_of_tokens_for_(ccat, tokens)
    influence_graph = [[0.0 for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i, j in zip(range(len(tokens)), range(len(tokens))):
        influence_graph[i][j] = vector.dot(ccat.get_query_for_(tokens[i]), ccat.get_key_for_(tokens[j]))
    for i in range(len(tokens)):
        row = influence_graph[i]
        max_val = max(row)
        exp_row = [CCAT.Const.E ** (val - max_val) for val in row]
        sum_exp = sum(exp_row)
        influence_graph[i] = [val / sum_exp for val in exp_row]
    tokens_influence = [0.0 for _ in range(len(tokens))]
    for i in range(len(influence_graph)):
        tokens_influence[i] = sum([influence_graph[i][j] for j in range(len(influence_graph))])
    sentence_meaning = Vector([0.0] * ccat.dim)
    cnt_n = Vector([0.0] * ccat.dim)
    cnt_m = 0.0
    for i in range(len(tokens)):
        weight = tokens_influence[i]
        weight_vec = [weight for _ in range(ccat.dim)]
        weighted_val = mexp.mul(weight_vec, tokens_meaning[i].components)
        cnt_n = mexp.add(cnt_n, Vector(weighted_val))
        cnt_m += weight
    result = mexp.div(cnt_n.components, cnt_m)
    sentence_meaning = Vector([r for r in result])
    return sentence_meaning

def get_meaning_of_sentence_at_(ccat: CCAT.CCATransformer, AtQ: Vector, tokens: List[str]) -> Vector:
    tokens_meaning = get_meaning_of_tokens_at_(ccat, AtQ, tokens)
    influence_graph = [[0.0 for _ in range(len(tokens))] for _ in range(len(tokens))]
    for i, j in zip(range(len(tokens)), range(len(tokens))):
        influence_graph[i][j] = vector.dot(ccat.get_query_for_(tokens[i]), ccat.get_key_for_(tokens[j]))
    for i in range(len(tokens)):
        row = influence_graph[i]
        max_val = max(row)
        exp_row = [CCAT.Const.E ** (val - max_val) for val in row]
        sum_exp = sum(exp_row)
        influence_graph[i] = [val / sum_exp for val in exp_row]
    tokens_influence = [0.0 for _ in range(len(tokens))]
    for i in range(len(influence_graph)):
        tokens_influence[i] = sum([influence_graph[i][j] for j in range(len(influence_graph))])
    sentence_meaning = Vector([0.0] * ccat.dim)
    cnt_n = Vector([0.0] * ccat.dim)
    cnt_m = 0.0
    for i in range(len(tokens)):
        weight = tokens_influence[i]
        weight_vec = [weight for _ in range(ccat.dim)]
        weighted_val = mexp.mul(weight_vec, tokens_meaning[i].components)
        cnt_n = mexp.add(cnt_n, Vector(weighted_val))
        cnt_m += weight
    result = mexp.div(cnt_n.components, cnt_m)
    sentence_meaning = Vector([r for r in result])
    return sentence_meaning

def softmax_choice_next_token_for_(ccat: CCAT.CCATransformer, tokens: List[str]) -> str:
    tokens_list, probabilities = softmax_choice_next_probability_for_(ccat, tokens)
    next_token = random.choices(tokens_list, probabilities)[0]
    return next_token

def softmax_choice_next_probability_for_(ccat: CCAT.CCATransformer, tokens: List[str]) -> str:
    sentence_meaning = get_meaning_of_sentence_for_(ccat, tokens)
    token_P = {}
    cnt = 0.0
    for token in ccat.get_tokens():
        dot_val = vector.dot(get_meaning_of_tokens_for_(ccat, tokens + [token])[-1], sentence_meaning)
        exponent = dot_val / ccat.temperature
        weight = CCAT.Const.E ** exponent
        token_P[token] = weight
        cnt += weight
    for token, weight in token_P.items():
        token_P[token] = weight / cnt if cnt != 0 else 0.0
    tokens_list = list(token_P.keys())
    probabilities = [i/sum(token_P.values()) for i in token_P.values()]
    return tokens_list, probabilities

def softmax_choice_next_token_at_(ccat: CCAT.CCATransformer, tokens: List[str], AtQ: Vector) -> str:
    tokens_list, probabilities = softmax_choice_next_probability_at_(ccat, tokens, AtQ)
    next_token = random.choices(tokens_list, probabilities)[0]
    return next_token

def softmax_choice_next_probability_at_(ccat: CCAT.CCATransformer, tokens: List[str], AtQ: Vector) -> str:
    sentence_meaning = get_meaning_of_sentence_at_(ccat, AtQ, tokens)
    token_P = {}
    cnt = 0.0
    for token in ccat.get_tokens():
        dot_val = vector.dot(get_meaning_of_tokens_at_(ccat, AtQ, tokens + [token])[-1], sentence_meaning)
        exponent = dot_val / ccat.temperature
        weight = CCAT.Const.E ** exponent
        token_P[token] = weight
        cnt += weight
    for token, weight in token_P.items():
        token_P[token] = weight / cnt if cnt != 0 else 0.0
    tokens_list = list(token_P.keys())
    probabilities = [i/sum(token_P.values()) for i in token_P.values()]
    return tokens_list, probabilities

def get_complate_for_(ccat: CCAT.CCATransformer, tokens: List[str]):
    sentence_meaning = get_meaning_of_sentence_for_(ccat, tokens[1:-1])
    token_meaning = get_meaning_of_tokens_for_(ccat, tokens)[-1]
    return vector.dot(sentence_meaning, token_meaning)

if __name__ == '__main__':
    import random
    ccat = NewCCATransformer(['苹果', '好吃', '香蕉', '橘子', '甜', '酸', '水果', '<end>'])
    start_tokens = [random.choice(ccat.get_tokens())] + [random.choice(ccat.get_tokens())] + [random.choice(ccat.get_tokens())]
    next_token = ''
    print(f'Start token: {start_tokens}')
    i = 0
    C = 1.0
    while next_token != "<end>":
        next_token = softmax_choice_next_token_for_(ccat, start_tokens)
        start_tokens.append(next_token)
        print(f'No. {i + 4} token: {next_token}', f'complate: {get_complate_for_(ccat, start_tokens)}')
        i += 1
    print('Generated token sequence:')
    print(start_tokens)
    print('with AtQ')
    start_tokens = start_tokens[:3]
    next_token = ''
    i = 0
    while next_token != "<end>":
        next_token = softmax_choice_next_token_at_(ccat, start_tokens, ccat.get_value_for_('苹果', ccat.get_query_for_('苹果')))
        start_tokens.append(next_token)
        print(f'No. {i + 4} token: {next_token}', f'complate: {get_complate_for_(ccat, start_tokens)}')
        i += 1
    print('Generated token sequence:')
    print(start_tokens)