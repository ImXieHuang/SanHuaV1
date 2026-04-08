import threading
from typing import Union, Any, List, Tuple
import sys
from pathlib import Path
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from Vector.vector import Vector

def is_vector(obj: Any) -> bool:
    return hasattr(obj, 'VECTORFLAG')

def is_list_like(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))

def contains_vectors(obj: Any) -> bool:
    if is_vector(obj):
        return True
    if is_list_like(obj):
        return any(contains_vectors(item) for item in obj)
    return False

def ensure_compatible(a: Any, b: Any) -> tuple:
    a_has_vectors = contains_vectors(a)
    b_has_vectors = contains_vectors(b)
    
    if a_has_vectors and b_has_vectors:
        if is_list_like(a) and is_list_like(b):
            if len(a) != len(b):
                max_len = max(len(a), len(b))
                a_list = list(a) + [1.0] * (max_len - len(a))
                b_list = list(b) + [1.0] * (max_len - len(b))
                return a_list, b_list
            return a, b
        elif is_vector(a) and is_vector(b):
            if a.dimension != b.dimension:
                max_dim = max(a.dimension, b.dimension)
                a_components = a.components + [1.0] * (max_dim - a.dimension)
                b_components = b.components + [1.0] * (max_dim - b.dimension)
                return Vector(a_components), Vector(b_components)
            return a, b
        elif is_vector(a) and is_list_like(b):
            b_vec = Vector(list(b))
            return ensure_compatible(a, b_vec)
        elif is_list_like(a) and is_vector(b):
            a_vec = Vector(list(a))
            return ensure_compatible(a_vec, b)
    
    elif a_has_vectors and isinstance(b, (int, float)):
        return a, b
    elif isinstance(a, (int, float)) and b_has_vectors:
        return a, b
    
    elif is_list_like(a) and is_list_like(b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a_list = list(a) + [1.0] * (max_len - len(a))
            b_list = list(b) + [1.0] * (max_len - len(b))
            return a_list, b_list
        return a, b
    
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a, b
    
    return a, b

def process_nested_operation(func: callable, a: Any, b: Any) -> Any:
    if is_list_like(a) and is_list_like(b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a_list = list(a) + [1.0] * (max_len - len(a))
            b_list = list(b) + [1.0] * (max_len - len(b))
            return [process_nested_operation(func, x, y) for x, y in zip(a_list, b_list)]
        return [process_nested_operation(func, x, y) for x, y in zip(a, b)]
    
    elif is_list_like(a) and not is_list_like(b):
        return [process_nested_operation(func, x, b) for x in a]
    
    elif not is_list_like(a) and is_list_like(b):
        return [process_nested_operation(func, a, x) for x in b]
    
    elif is_vector(a) and is_vector(b):
        if a.dimension != b.dimension:
            max_dim = max(a.dimension, b.dimension)
            a_components = a.components + [1.0] * (max_dim - a.dimension)
            b_components = b.components + [1.0] * (max_dim - b.dimension)
            return Vector([func(x, y) for x, y in zip(a_components, b_components)])
        return Vector([func(x, y) for x, y in zip(a.components, b.components)])
    
    elif is_vector(a) and isinstance(b, (int, float)):
        return Vector([func(x, b) for x in a.components])
    
    elif isinstance(a, (int, float)) and is_vector(b):
        return Vector([func(a, x) for x in b.components])
    
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return func(a, b)
    
    else:
        try:
            return func(a, b)
        except Exception:
            raise TypeError(f"Unsupported operand types: {type(a)} and {type(b)}")

def mul(a: Union[Vector, int, float, list], b: Union[Vector, int, float, list] = 1) -> Union[Vector, float, list]:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) + str(b)
    
    def multiply_func(x, y):
        return x * y
    
    return process_nested_operation(multiply_func, a, b)

def div(a: Union[Vector, int, float, list], b: Union[Vector, int, float, list] = 1, miny: float = 1e-10) -> Union[Vector, float, list]:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) + str(b)
    
    def divide_func(x, y):
        abs_y = abs(y)
        if abs_y <= miny:
            y = miny
        return x / y
    
    return process_nested_operation(divide_func, a, b)

def add(a: Union[Vector, int, float, list], b: Union[Vector, int, float, list] = 0) -> Union[Vector, float, list]:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) + str(b)

    def add_func(x, y):
        return x + y
    
    return process_nested_operation(add_func, a, b)

def sub(a: Union[Vector, int, float, list], b: Union[Vector, int, float, list] = 0) -> Union[Vector, float, list]:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) + str(b)
    
    def subtract_func(x, y):
        return x - y
    
    return process_nested_operation(subtract_func, a, b)

def iterate(func: callable, items: list) -> Any:
    if len(items) == 0:
        return 0.0
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return func(items[0], items[1])
    else:
        return func(items[0], iterate(func, items[1:]))

class LinearAlgebra:
    def __init__(self):
        pass

    def is_matrix(self, obj):
        if not obj or not is_list_like(obj):
            return False
        if not all(isinstance(row, list) for row in obj):
            return False
        col_count = len(obj[0])
        if not all(len(row) == col_count for row in obj):
            return False
        return True
    
    def dim(self, obj):
        if is_vector(obj):
            return (obj.dimension,)
        elif self.is_matrix(obj):
            return (len(obj), len(obj[0]))
        else:
            raise ValueError("obj must be Vector or Matrix")
        
    def mul(self, a, b):
        dim_a = self.dim(a)
        dim_b = self.dim(b)
        
        if len(dim_a) == 1 and len(dim_b) == 1:
            if dim_a[0] != dim_b[0]:
                raise ValueError("Vector dimensions must match for dot product")
            a_comp = a.components if is_vector(a) else a
            b_comp = b.components if is_vector(b) else b
            return sum(ai * bi for ai, bi in zip(a_comp, b_comp))
        
        elif len(dim_a) == 2 and len(dim_b) == 2:
            if dim_a[1] != dim_b[0]:
                raise ValueError(f"Matrix dimensions incompatible: {dim_a} and {dim_b}")
            result = []
            for i in range(dim_a[0]):
                row = []
                for j in range(dim_b[1]):
                    val = 0
                    for k in range(dim_a[1]):
                        val = add(val, mul(a[i][k], b[k][j]))
                    row.append(val)
                result.append(row)
            return result
        
        elif len(dim_a) == 2 and len(dim_b) == 1:
            if dim_a[1] != dim_b[0]:
                raise ValueError("Matrix and vector dimensions incompatible")
            result = []
            b_comp = b.components if is_vector(b) else b
            for i in range(dim_a[0]):
                val = 0
                for k in range(dim_a[1]):
                    val = add(val, mul(a[i][k], b_comp[k]))
                result.append(val)
            return Vector(result) if is_vector(b) else result
        
        elif len(dim_a) == 1 and len(dim_b) == 2:
            if dim_a[0] != dim_b[0]:
                raise ValueError("Vector and matrix dimensions incompatible")
            result = []
            a_comp = a.components if is_vector(a) else a
            for j in range(dim_b[1]):
                val = 0
                for k in range(dim_a[0]):
                    val = add(val, mul(a_comp[k], b[k][j]))
                result.append(val)
            return Vector(result) if is_vector(a) else result
        
        else:
            raise ValueError("Unsupported operand types")

class ThreadManager:
    def __init__(self):
        self._threads = []
        self._results = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition()

    def push(self, function, val):
        def target(func, args, tid):
            try:
                result = func(*args)
            except Exception as e:
                result = None
            with self._lock:
                self._results[tid] = result
            with self._condition:
                self._condition.notify_all()

        thread = threading.Thread(target=target, args=(function, val, id(threading.current_thread())))
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def waite(self, *threads):
        results = []
        for t in threads:
            if not isinstance(t, list):
                t = [t]
            with self._lock:
                results.append(self._results.get(id(t), None))
        return [i for i in results]

if __name__ == '__main__':
    la = LinearAlgebra()
    print(la.mul(Vector([1.0,0.0]), [[0.0,1.0],[1.0,0.0]]))