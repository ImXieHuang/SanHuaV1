import math
from .vector import Vector
from typing import Union

def _validate_vectors(v1: Vector, v2: Vector):
    if v1.dimension != v2.dimension:
        raise ValueError('Vectors must have the same dimension')

def add(v1: Vector, v2: Vector) -> Vector:
    _validate_vectors(v1, v2)
    return Vector([x + y for x, y in zip(v1.components, v2.components)])

def sub(v1: Vector, v2: Vector) -> Vector:
    _validate_vectors(v1, v2)
    return Vector([x - y for x, y in zip(v1.components, v2.components)])

def mul(v1: Vector, v2: Vector) -> Vector:
    _validate_vectors(v1, v2)
    return Vector([x * y for x, y in zip(v1.components, v2.components)])

def div(v1: Vector, v2: Vector) -> Vector:
    _validate_vectors(v1, v2)
    if any((y == 0 for y in v2.components)):
        raise ValueError('Cannot divide by zero in vector components')
    return Vector([x / y for x, y in zip(v1.components, v2.components)])

def dot(v1: Vector, v2: Vector) -> Union[float, float]:
    _validate_vectors(v1, v2)
    result = sum((x * y for x, y in zip(v1.components, v2.components)))
    if all((isinstance(x, float) for x in v1.components)) and all((isinstance(x, float) for x in v2.components)):
        return int(result * 1000) / 1000
    return result

def cross(v1: Vector, v2: Vector) -> Vector:
    if v1.dimension != 3 or v2.dimension != 3:
        raise ValueError('Cross product is only defined for 3-dimensional vectors')
    a = v1.components
    b = v2.components
    cross_x = a[1] * b[2] - a[2] * b[1]
    cross_y = a[2] * b[0] - a[0] * b[2]
    cross_z = a[0] * b[1] - a[1] * b[0]
    return Vector([cross_x, cross_y, cross_z])

def abs_vector(v: Vector):
    abs_list = []
    for i in v.components:
        if isinstance(i, float):
            abs_list.append(abs(i.numerator) / abs(i.denominator))
        else:
            abs_list.append(abs(i))
    return Vector(abs_list)

def compare(v1: Vector, v2: Vector) -> Union[float, float]:
    diff_vector = sub(v1, v2)
    abs_diff = abs_vector(diff_vector)
    if all((isinstance(x, float) for x in abs_diff.components)):
        result = 1.0
        for comp in abs_diff.components:
            result = result * comp
        return result
    else:
        result = 1.0
        for comp in abs_diff.components:
            result *= comp
        return result

def magnitude(v: Vector) -> Union[float, float]:
    squared_sum = sum((x * x for x in v.components))
    if all((isinstance(x, float) for x in v.components)):
        return math.sqrt(float(squared_sum))
    else:
        return math.sqrt(squared_sum)

def normalize(v: Vector) -> Vector:
    if all((x == 0 for x in v.components)):
        return v
    mag = magnitude(v)
    if all((isinstance(x, float) for x in v.components)):
        normalized_components = [x / mag for x in v.components]
    else:
        normalized_components = [x / mag for x in v.components]
    return Vector(normalized_components)

def is_unit_vector(v: Vector, tolerance: float=1e-10) -> bool:
    mag = magnitude(v)
    return abs(mag - 1.0) < tolerance