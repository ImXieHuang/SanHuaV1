import sys
import math
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)
from Vector.vector import Vector
from Vector import dot
from mathexpand import mul, div, add, sub

@dataclass(frozen=True)
class Const:
    E = 2.718281828459065
    Pi = 3.1415926535897927
    INF = float('inf')

class CCATransformer:

    def __init__(self, database: Dict[str, Dict[Vector, Vector]], temperature: float=1.0, dim=8):
        self.database = database
        self.temperature = temperature
        self.dim = dim
        self._validate_database()

    def _validate_database(self):
        for token, query_dict in self.database.items():
            if not isinstance(token, str):
                raise TypeError('The type of the token must be string.')
            if not isinstance(query_dict, dict):
                raise TypeError('The type of the QueryDatabase must be dictionary.')
            for query_vec, value_vec in query_dict.items():
                if not hasattr(query_vec, 'VECTORFLAG'):
                    raise TypeError('The type of the Query must be Vector.')
                if not hasattr(value_vec, 'VECTORFLAG'):
                    raise TypeError('The type of the Val must be Vector.')

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

    def SoftInjection_to_(self, token: str, Query: Vector, Value: Vector, const: float=Const.E):
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        injection = self.database[token]
        if not injection:
            self.database[token] = {Query: Value}
            return
        query_list = list(injection.keys())
        value_list = list(injection.values())
        influence_graph = []
        total_weight = 0.0
        for query_vec in query_list:
            dot_val = int(dot(query_vec, Query) * 1000) / 1000
            exponent = div(dot_val, mul(self.temperature, self.dim))
            weight = const ** exponent
            influence_graph.append(weight)
            total_weight = add(total_weight, weight)
        if total_weight != 0:
            influence_graph = [div(weight, total_weight) for weight in influence_graph]
        updated_dict = {}
        for i, (query_vec, value_vec) in enumerate(zip(query_list, value_list)):
            delta = mul(sub(Value, value_vec).components, influence_graph[i])
            new_value = add(value_vec, Vector(delta))
            updated_dict[query_vec] = new_value
        updated_dict[Query] = Value
        self.database[token] = updated_dict

    def SoftQuery(self, token: str, Query: Vector, const: float=Const.E) -> Vector:
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        query_dict = self.database[token]
        if not query_dict:
            return Vector([0.0] * self.dim)
        cnt_n = Vector([0.0] * self.dim)
        cnt_m = 0.0
        for i in query_dict:
            dot_val = int(dot(i, Query) * 1000) / 1000
            exponent = div(dot_val, mul(self.temperature, self.dim))
            weight = const ** exponent
            weight_vec = [weight for _ in range(self.dim)]
            weighted_val = mul(weight_vec, query_dict[i].components)
            cnt_n = add(cnt_n, Vector(weighted_val))
            cnt_m = add(cnt_m, weight)
        if cnt_m == 0:
            return Vector([c for c in cnt_n.components])
        result = div(cnt_n.components, cnt_m)
        return Vector([r for r in result])

    def DeSoftQuery(self, token: str, Val: Vector, const: float=Const.E) -> Vector:
        if token not in self.database:
            raise KeyError(f"token '{token}' not found in database")
        query_dict = self.database[token]
        if not query_dict:
            return Vector([0.0] * self.dim)
        cnt_n = Vector([0.0] * self.dim)
        cnt_m = 0.0
        for i in query_dict:
            dot_val = dot(mul(sub(Val, query_dict[i]), sub(Val, query_dict[i])), Vector([1.0] * self.dim))
            exponent = div(dot_val, mul(self.temperature, self.dim))
            weight = const ** exponent
            weight_vec = [weight for _ in range(self.dim)]
            weighted_val = mul(weight_vec, i.components)
            cnt_n = add(cnt_n, Vector(weighted_val))
            cnt_m = add(cnt_m, weight)
        if cnt_m == 0:
            return Vector([c for c in cnt_n.components])
        result = div(cnt_n.components, cnt_m)
        return Vector([r for r in result])

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
        return f'ContextAwareTransformer(database_size={len(self.database)}, temperature={self.temperature})'

    def __str__(self):
        summary = []
        for token in self.database:
            summary.append(f'{token}: {len(self.database[token])} queries')
        return '\n'.join(summary)
if __name__ == '__main__':
    print(f'const e: {Const.E}')
    print(f'const pi: {Const.Pi}\n')
    ccat = CCATransformer({'苹果': {Vector([1, 0, 0, 0, 1, 0, 0, 0]): Vector([1, 1, 1, 1, 1, 1, 1, 1]), Vector([0, 0, 1, 0, 0, 0, 1, 0]): Vector([2, 2, 2, 2, 2, 2, 2, 2])}, '香蕉': {Vector([0, 1, 0, 0, 0, 1, 0, 0]): Vector([2, 2, 2, 2, 2, 2, 2, 2])}}, temperature=Const.E)
    big_Q = ccat.get_query_for_('苹果')
    print(f"big_Q for '苹果': {big_Q} \n")
    print(f"Keys for '苹果': {ccat.get_key_for_('苹果')}")
    print(f"Values for '苹果': {ccat.get_value_for_('苹果', big_Q)}")
    print(f"Query for '苹果': {ccat.get_query_for_('苹果')}")
    print(f"Keys for '香蕉': {ccat.get_key_for_('香蕉')}")
    print(f"Values for '香蕉': {ccat.get_value_for_('香蕉', big_Q)}")
    print(f"Query for '香蕉': {ccat.get_query_for_('香蕉')}")