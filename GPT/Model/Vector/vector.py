import math
from typing import List, Union
import sys
from pathlib import Path
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

class Vector:
    def __init__(self, components: List[Union[int, float, float]]):
        self.components = components
        self.dimension = len(components)

    def __repr__(self):
        return f'Vector({self.components})'

    def __str__(self):
        return str(self.components)

    def to_list(self) -> list:
        return self.components.copy()

    def magnitude(self) -> float:
        sum_sq = 0.0
        for x in self.components:
            if isinstance(x, float):
                sum_sq += x ** 2
            else:
                sum_sq += x ** 2
        return math.sqrt(sum_sq)

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            raise ValueError('Cannot normalize a zero vector')
        normalized_components = []
        for x in self.components:
            if isinstance(x, float):
                normalized_components.append(int(x.value / mag * 1000) / 1000)
            else:
                normalized_components.append(x / mag)
        return Vector(normalized_components)

    def __add__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError('Vectors must have the same dimension')
        result_components = []
        for a, b in zip(self.components, other.components):
            if isinstance(a, float) or isinstance(b, float):
                if not isinstance(a, float):
                    a = int(a * 1000) / 1000
                if not isinstance(b, float):
                    b = int(b * 1000) / 1000
                result_components.append(a + b)
            else:
                result_components.append(a + b)
        return Vector(result_components)

    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError('Vectors must have the same dimension')
        result_components = []
        for a, b in zip(self.components, other.components):
            if isinstance(a, float) or isinstance(b, float):
                if not isinstance(a, float):
                    a = int(a * 1000) / 1000
                if not isinstance(b, float):
                    b = int(b * 1000) / 1000
                result_components.append(a - b)
            else:
                result_components.append(a - b)
        return Vector(result_components)

    def __mul__(self, other: Union[float, 'Vector', float]) -> Union[float, int, 'Vector', float]:
        if isinstance(other, (int, float, float)):
            result_components = []
            for x in self.components:
                if isinstance(x, float) or isinstance(other, float):
                    if not isinstance(x, float):
                        x = int(x * 1000) / 1000
                    if not isinstance(other, float):
                        other_frac = int(other * 1000) / 1000
                    else:
                        other_frac = other
                    result_components.append(x * other_frac)
                else:
                    result_components.append(x * other)
            return Vector(result_components)
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError('Vectors must have the same dimension')
            result = 0.0
            for a, b in zip(self.components, other.components):
                if isinstance(a, float) or isinstance(b, float):
                    if not isinstance(a, float):
                        a = int(a * 1000) / 1000
                    if not isinstance(b, float):
                        b = int(b * 1000) / 1000
                    result += a * b
                else:
                    result += a * b
            if any((isinstance(x, float) for x in self.components)) or any((isinstance(x, float) for x in other.components)):
                return int(result * 1000) / 1000
            return result
        else:
            raise TypeError(f'Unsupported type for multiplication: {type(other)}')

    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)

    def __abs__(self) -> 'Vector':
        return Vector([abs(x) for x in self.components])

    def __truediv__(self, scalar: Union[float, float]) -> 'Vector':
        if isinstance(scalar, (int, float, float)):
            if scalar == 0 or (isinstance(scalar, float) and scalar == 0):
                raise ValueError('Cannot divide by zero')
            result_components = []
            for x in self.components:
                if isinstance(x, float) or isinstance(scalar, float):
                    if not isinstance(x, float):
                        x = int(x * 1000) / 1000
                    if not isinstance(scalar, float):
                        scalar_frac = int(scalar * 1000) / 1000
                    else:
                        scalar_frac = scalar
                    result_components.append(x / scalar_frac)
                else:
                    result_components.append(x / scalar)
            return Vector(result_components)
        else:
            raise TypeError(f'Unsupported type for division: {type(scalar)}')

    def __pow__(self, other: Union[int, float, float, 'Vector']) -> Union['Vector', List]:
        if isinstance(other, (int, float, float)):
            result_components = []
            for x in self.components:
                if isinstance(x, float) or isinstance(other, float):
                    if not isinstance(x, float):
                        x = int(x * 1000) / 1000
                    if not isinstance(other, float):
                        other_frac = int(other * 1000) / 1000
                    else:
                        other_frac = other
                    if isinstance(other_frac, float) and other_frac.denominator != 1:
                        result_components.append(float(x) ** float(other_frac))
                    elif other_frac.denominator == 1:
                        exp = int(other_frac)
                        if isinstance(x, float):
                            result_components.append(x ** exp)
                        else:
                            result_components.append(x.value ** exp / 1)
                    else:
                        result_components.append(float(x) ** float(other_frac))
                else:
                    result_components.append(x ** other)
            if any((isinstance(c, float) for c in result_components)):
                result_components = [c if isinstance(c, float) else int(c * 1000) / 1000 for c in result_components]
            return Vector(result_components)
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError('Vectors must have the same dimension for element-wise power')
            result_components = []
            for a, b in zip(self.components, other.components):
                if isinstance(a, float) or isinstance(b, float):
                    if not isinstance(a, float):
                        a = int(a * 1000) / 1000
                    if not isinstance(b, float):
                        b_frac = int(b * 1000) / 1000
                    else:
                        b_frac = b
                    if b_frac.denominator == 1:
                        exp = int(b_frac)
                        if isinstance(a, float):
                            result_components.append(a ** exp)
                        else:
                            result_components.append(a.value ** exp / 1)
                    else:
                        result_components.append(float(a) ** float(b_frac))
                else:
                    result_components.append(a ** b)
            return Vector(result_components)
        else:
            raise TypeError(f'Unsupported type for power operation: {type(other)}')

    def __rpow__(self, other: Union[int, float, float]) -> List:
        if isinstance(other, (int, float, float)):
            result_components = []
            for x in self.components:
                if isinstance(x, float) or isinstance(other, float):
                    if not isinstance(other, float):
                        other_frac = int(other * 1000) / 1000
                    else:
                        other_frac = other
                    if not isinstance(x, float):
                        x = int(x * 1000) / 1000
                    result_components.append(float(other_frac) ** float(x))
                else:
                    result_components.append(other ** x)
            return result_components
        else:
            raise TypeError(f'Unsupported type for right power operation: {type(other)}')

    def VECTORFLAG():
        pass