import threading
from typing import Union, Any, List, Tuple
import sys
from pathlib import Path
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from Vector.vector import Vector
from fraction import fraction

def is_vector(obj: Any) -> bool:
    return hasattr(obj, 'VECTORFLAG')

def is_fraction(obj: Any) -> bool:
    return isinstance(obj, fraction)

def is_list_like(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))

def contains_vectors(obj: Any) -> bool:
    if is_vector(obj):
        return True
    if is_list_like(obj):
        return any(contains_vectors(item) for item in obj)
    return False

def contains_fractions(obj: Any) -> bool:
    if is_fraction(obj):
        return True
    if is_vector(obj):
        return any(is_fraction(x) for x in obj.components)
    if is_list_like(obj):
        return any(contains_fractions(item) for item in obj)
    return False

def to_fraction(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return fraction(int(value * 1000), 1000)
    elif is_list_like(value):
        return [to_fraction(item) for item in value]
    elif is_vector(value):
        components = []
        for comp in value.components:
            if is_fraction(comp):
                components.append(comp)
            else:
                components.append(fraction(int(comp * 1000), 1000))
        return Vector(components)
    else:
        return value

def ensure_compatible(a: Any, b: Any) -> tuple:
    a_has_vectors = contains_vectors(a)
    b_has_vectors = contains_vectors(b)
    a_has_fractions = contains_fractions(a)
    b_has_fractions = contains_fractions(b)
    
    if a_has_fractions or b_has_fractions:
        a = to_fraction(a)
        b = to_fraction(b)
    
    if a_has_vectors and b_has_vectors:
        if is_list_like(a) and is_list_like(b):
            if len(a) != len(b):
                max_len = max(len(a), len(b))
                if a_has_fractions:
                    a_list = list(a) + [fraction(1, 1)] * (max_len - len(a))
                    b_list = list(b) + [fraction(1, 1)] * (max_len - len(b))
                else:
                    a_list = list(a) + [1.0] * (max_len - len(a))
                    b_list = list(b) + [1.0] * (max_len - len(b))
                return a_list, b_list
            return a, b
        elif is_vector(a) and is_vector(b):
            if a.dimension != b.dimension:
                max_dim = max(a.dimension, b.dimension)
                if a_has_fractions:
                    a_components = a.components + [fraction(1, 1)] * (max_dim - a.dimension)
                    b_components = b.components + [fraction(1, 1)] * (max_dim - b.dimension)
                else:
                    a_components = a.components + [1.0] * (max_dim - a.dimension)
                    b_components = b.components + [1.0] * (max_dim - b.dimension)
                return Vector(a_components), Vector(b_components)
            return a, b
        elif is_vector(a) and is_list_like(b):
            if a_has_fractions:
                b_vec = Vector([fraction(x, 1) if isinstance(x, (int, float)) else x for x in list(b)])
            else:
                b_vec = Vector(list(b))
            return ensure_compatible(a, b_vec)
        elif is_list_like(a) and is_vector(b):
            if b_has_fractions:
                a_vec = Vector([fraction(x, 1) if isinstance(x, (int, float)) else x for x in list(a)])
            else:
                a_vec = Vector(list(a))
            return ensure_compatible(a_vec, b)
    
    elif a_has_vectors and isinstance(b, (int, float, fraction)):
        return a, b
    elif isinstance(a, (int, float, fraction)) and b_has_vectors:
        return a, b
    
    elif is_list_like(a) and is_list_like(b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            if a_has_fractions or b_has_fractions:
                a_list = list(a) + [fraction(1, 1)] * (max_len - len(a))
                b_list = list(b) + [fraction(1, 1)] * (max_len - len(b))
            else:
                a_list = list(a) + [1.0] * (max_len - len(a))
                b_list = list(b) + [1.0] * (max_len - len(b))
            return a_list, b_list
        return a, b
    
    elif isinstance(a, (int, float, fraction)) and isinstance(b, (int, float, fraction)):
        return a, b
    
    return a, b

def process_nested_operation(func: callable, a: Any, b: Any) -> Any:
    if is_list_like(a) and is_list_like(b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            if contains_fractions(a) or contains_fractions(b):
                a_list = list(a) + [fraction(1, 1)] * (max_len - len(a))
                b_list = list(b) + [fraction(1, 1)] * (max_len - len(b))
            else:
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
            if contains_fractions(a) or contains_fractions(b):
                a_components = a.components + [fraction(1, 1)] * (max_dim - a.dimension)
                b_components = b.components + [fraction(1, 1)] * (max_dim - b.dimension)
            else:
                a_components = a.components + [1.0] * (max_dim - a.dimension)
                b_components = b.components + [1.0] * (max_dim - b.dimension)
            return Vector([func(x, y) for x, y in zip(a_components, b_components)])
        return Vector([func(x, y) for x, y in zip(a.components, b.components)])
    
    elif is_vector(a) and isinstance(b, (int, float, fraction)):
        return Vector([func(x, b) for x in a.components])
    
    elif isinstance(a, (int, float, fraction)) and is_vector(b):
        return Vector([func(a, x) for x in b.components])
    
    elif isinstance(a, fraction) or isinstance(b, fraction):
        if isinstance(a, (int, float)):
            a = fraction(int(a * 1000), 1000)
        if isinstance(b, (int, float)):
            b = fraction(int(b * 1000), 1000)
        return func(a, b)
    
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return func(a, b)
    
    else:
        try:
            return func(a, b)
        except Exception:
            raise TypeError(f"Unsupported operand types: {type(a)} and {type(b)}")

def mul(a: Union[Vector, int, float, fraction, list], b: Union[Vector, int, float, fraction, list] = 1) -> Union[Vector, float, fraction, list]:
    def multiply_func(x, y):
        if isinstance(x, fraction) or isinstance(y, fraction):
            if not isinstance(x, fraction):
                x = fraction(int(x * 1000), 1000)
            if not isinstance(y, fraction):
                y = fraction(int(y * 1000), 1000)
            return x * y
        return x * y
    
    return process_nested_operation(multiply_func, a, b)

def div(a: Union[Vector, int, float, fraction, list], b: Union[Vector, int, float, fraction, list] = 1, miny: float = 1e-10) -> Union[Vector, float, fraction, list]:
    def divide_func(x, y):
        abs_y = abs(y.value if isinstance(y, fraction) else y)
        if abs_y <= miny:
            y = fraction(int(miny * 1000), 1000) if isinstance(x, fraction) else miny
        
        if isinstance(x, fraction) or isinstance(y, fraction):
            if not isinstance(x, fraction):
                x = fraction(int(x * 1000), 1000)
            if not isinstance(y, fraction):
                y = fraction(int(y * 1000), 1000)
            return x / y
        return x / y
    
    return process_nested_operation(divide_func, a, b)

def add(a: Union[Vector, int, float, fraction, list], b: Union[Vector, int, float, fraction, list] = 0) -> Union[Vector, float, fraction, list]:
    def add_func(x, y):
        if isinstance(x, fraction) or isinstance(y, fraction):
            if not isinstance(x, fraction):
                x = fraction(int(x * 1000), 1000)
            if not isinstance(y, fraction):
                y = fraction(int(y * 1000), 1000)
            return x + y
        return x + y
    
    return process_nested_operation(add_func, a, b)

def sub(a: Union[Vector, int, float, fraction, list], b: Union[Vector, int, float, fraction, list] = 0) -> Union[Vector, float, fraction, list]:
    def subtract_func(x, y):
        if isinstance(x, fraction) or isinstance(y, fraction):
            if not isinstance(x, fraction):
                x = fraction(int(x * 1000), 1000)
            if not isinstance(y, fraction):
                y = fraction(int(y * 1000), 1000)
            return x - y
        return x - y
    
    return process_nested_operation(subtract_func, a, b)

def iterate(func: callable, items: list) -> Any:
    if len(items) == 0:
        return fraction(0, 1) if contains_fractions(items) else 0.0
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return func(items[0], items[1])
    else:
        return func(items[0], iterate(func, items[1:]))
    
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
    from fraction import fraction
    
    print(is_vector(Vector([1,2,3])))
    print(is_vector(6))

    print("Nested list with vectors and fractions:")
    vec_list1 = [Vector([fraction(1, 1), fraction(2, 1)]), Vector([3, 4])]
    vec_list2 = [Vector([2, 3]), Vector([fraction(4, 1), fraction(5, 1)])]
    print(mul(vec_list1, vec_list2))
    print(add(vec_list1, fraction(1, 1)))
    print(mul(vec_list1, 2))
    
    print("Mixed nested structures with fractions:")
    mixed1 = [Vector([1, fraction(2, 1)]), [Vector([3, 4]), Vector([5, 6])]]
    mixed2 = [2, [fraction(3, 1), 4]]
    print(mul(mixed1, mixed2))
    
    print("Deep nesting with fractions:")
    deep1 = [[Vector([1, 2]), Vector([fraction(3, 1), 4])], [Vector([5, 6])]]
    deep2 = [[2, fraction(3, 1)], [4]]
    print(add(deep1, deep2))
    
    print("Pure fraction operations:")
    print(add(fraction(1, 2), fraction(1, 3)))
    print(mul(fraction(2, 3), 1.5))
    print(div(fraction(3, 4), 0.25))