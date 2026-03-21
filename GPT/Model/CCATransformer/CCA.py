import sys
import math
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from Vector.vector import Vector
from Vector import dot
from fraction import fraction
from mathexpand import mul, div, add, sub

@dataclass(frozen=True)
class Const():
    E = fraction(13580623, 4996032) # 10^-15
    Pi = fraction(80143857, 25510582) # 10^-15
    INF = float('inf')

class CCATransformer():
    def __init__(self, database: Dict[str, Dict[Vector, Vector]], temperature: fraction = fraction(1, 1), dim = 8):
        self.database = database
        self.temperature = temperature
        self.dim = dim
        
        self._validate_database()
    
    def _validate_database(self):
        for token, query_dict in self.database.items():
            if not isinstance(token, str):
                raise TypeError("The type of the token must be string.")
            if not isinstance(query_dict, dict):
                raise TypeError("The type of the QueryDatabase must be dictionary.")
            for query_vec, value_vec in query_dict.items():
                if not hasattr(query_vec, "VECTORFLAG"):
                    raise TypeError("The type of the Query must be Vector.")
                if not hasattr(value_vec, "VECTORFLAG"):
                    raise TypeError("The type of the Val must be Vector.")

    def get_key_for_(self, token: str):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        return div(sum([v for v in self.database[token].values()], Vector([0.0] * self.dim)), len(list(self.database[token].values())))
    
    def get_value_for_(self, token: str, big_Q: Vector):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        return self.SoftQuery(token, big_Q, self.temperature)
    
    def get_query_for_(self, token: str):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        return self.DeSoftQuery(token, self.get_key_for_(token), self.temperature)
    
    def query_best_token_for_(self, TokenVec: Vector, big_Q: Vector) -> str:
        voronoi = {}
        for token in self.database:
            voronoi[self.get_value_for_(token, big_Q)] = token
        
        best_loss = -float('inf')
        best_token = ""
        for token in voronoi:
            if dot(token, TokenVec) > best_loss:
                best_loss = dot(token, TokenVec)
                best_token = voronoi[token]
        
        return best_token
    
    def SoftInjection_to_(self, token: str, Query: Vector, Value: Vector, T: fraction = fraction(3, 2), const: fraction = Const.E):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        
        injection = self.database[token]
        influence_graph = [0.0]*len(list(injection.keys()))
        maxn = fraction(100, 1)
        minn = fraction(-100, 1)
        cnt = fraction(0, 1)

        for i in injection:
            dot_val = fraction(int(dot(i, Query) * 1000), 1000)
            exponent = div(dot_val, mul(T, self.dim))
            
            exp_sub_max = sub(exponent, maxn)
            e_pow = const ** (exp_sub_max.value)
            denom1 = sub(fraction(1, 1), e_pow)
            first_term = div(exp_sub_max, denom1)
            exponent = add(first_term, maxn)
            
            exp_add_min = add(exponent, minn)
            e_pow2 = const ** (-exp_add_min.value)
            denom2 = sub(fraction(1, 1), e_pow2)
            second_term = div(exp_add_min, denom2)
            exponent = sub(second_term, minn)
            
            weight = const ** exponent.value
            influence_graph[i] = weight
            cnt = add(cnt, weight)

        influence_graph = [influence_graph[i] / cnt.value for i in range(len(influence_graph))]

        delat_graph = [(Value - list(injection.values())[i]) * influence_graph[i] for i in range(len(list(injection.values())))]

        injection = [influence_graph[i] * delat_graph[i] for i in range(len(influence_graph))]

        self.database[token] = injection

    def SoftQuery(self, token: str, Query: Vector, T: fraction = fraction(3, 2), const: fraction = Const.E) -> Vector:
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        
        query_dict = self.database[token]
        if not query_dict:
            return Vector([0.0] * self.dim)
        
        cnt_n = Vector([fraction(0, 1)] * self.dim)
        cnt_m = fraction(0, 1)

        for i in query_dict:
            dot_val = fraction(int(dot(i, Query) * 1000), 1000)
            exponent = div(dot_val, mul(T, self.dim))
            
            weight = const.value ** exponent.value
            weight_vec = [weight for _ in range(self.dim)]
            weighted_val = mul(weight_vec, query_dict[i].components)
            cnt_n = add(cnt_n, Vector(weighted_val))
            cnt_m = add(cnt_m, weight)
        
        if cnt_m.value == 0:
            return Vector([c.value for c in cnt_n.components])
        
        result = div(cnt_n.components, cnt_m)
        return Vector([r.value for r in result])
    
    def DeSoftQuery(self, token: str, Val: Vector, T: fraction = fraction(3, 2), const: fraction = Const.E) -> Vector:
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        
        query_dict = self.database[token]
        if not query_dict:
            return Vector([0.0] * self.dim)
        
        cnt_n = Vector([fraction(0, 1)] * self.dim)
        cnt_m = fraction(0, 1)

        for i in query_dict:
            dot_val = dot(mul(sub(Val, query_dict[i]), sub(Val, query_dict[i])), Vector([1.0] * self.dim))
            exponent = div(dot_val, mul(T, self.dim))
            
            weight = const.value ** exponent
            weight_vec = [weight for _ in range(self.dim)]
            weighted_val = mul(weight_vec, i.components)
            cnt_n = add(cnt_n, Vector(weighted_val))
            cnt_m = add(cnt_m, weight)
        
        if cnt_m.value == 0:
            return Vector([c.value for c in cnt_n.components])
        
        result = div(cnt_n.components, cnt_m)
        return Vector([r.value for r in result])

    def get_tokens(self):
        return list(self.database.keys())
    
    def get_queries_for_token(self, token: str):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        return list(self.database[token].keys())
    
    def get_values_for_token(self, token: str):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        return list(self.database[token].values())
    
    def __repr__(self):
        return f"ContextAwareTransformer(database_size={len(self.database)}, temperature={self.temperature})"
    
    def __str__(self):
        summary = []
        for token in self.database:
            summary.append(f"{token}: {len(self.database[token])} queries")
        return "\n".join(summary)
    
if __name__ == "__main__":
    print(f"const e: {Const.E.value}")
    print(f"const pi: {Const.Pi.value}\n")
    
    cat = CCATransformer(
        {
            "苹果": {
                Vector([1,0,0,0,1,0,0,0]): Vector([1,1,1,1,1,1,1,1]),
                Vector([0,0,1,0,0,0,1,0]): Vector([2,2,2,2,2,2,2,2])
            },
            "香蕉": {
                Vector([0,1,0,0,0,1,0,0]): Vector([2,2,2,2,2,2,2,2])
            }
        },
        temperature=Const.E
    )

    big_Q = cat.get_query_for_("苹果")
    print(f"big_Q for '苹果': {big_Q} \n")

    print(f"Keys for '苹果': {cat.get_key_for_('苹果')}")
    print(f"Values for '苹果': {cat.get_value_for_('苹果', big_Q)}")
    print(f"Query for '苹果': {cat.get_query_for_('苹果')}")
    print(f"Keys for '香蕉': {cat.get_key_for_('香蕉')}")
    print(f"Values for '香蕉': {cat.get_value_for_('香蕉', big_Q)}")
    print(f"Query for '香蕉': {cat.get_query_for_('香蕉')}")

    print(f"\nBest token for Vector([1.5]*8): {cat.query_best_token_for_(Vector([1.5]*8), big_Q)}")
