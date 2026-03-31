import math
from typing import List, Union
import sys
from pathlib import Path
udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

class Vector:
    def __init__(self, components: List[Union[int, float]]):
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
            sum_sq += x ** 2
        return math.sqrt(sum_sq)

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            raise ValueError('Cannot normalize a zero vector')
        normalized_components = []
        for x in self.components:
            normalized_components.append(x / mag)
        return Vector(normalized_components)

    def __add__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError('Vectors must have the same dimension')
        result_components = []
        for a, b in zip(self.components, other.components):
            result_components.append(a + b)
        return Vector(result_components)

    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError('Vectors must have the same dimension')
        result_components = []
        for a, b in zip(self.components, other.components):
            result_components.append(a - b)
        return Vector(result_components)

    def __mul__(self, other: Union[float, 'Vector']) -> Union[float, int, 'Vector']:
        if isinstance(other, (int, float)):
            result_components = []
            for x in self.components:
                result_components.append(x * other)
            return Vector(result_components)
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError('Vectors must have the same dimension')
            result = 0.0
            for a, b in zip(self.components, other.components):
                result += a * b
            return result
        else:
            raise TypeError(f'Unsupported type for multiplication: {type(other)}')

    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)

    def __abs__(self) -> 'Vector':
        return Vector([abs(x) for x in self.components])

    def __truediv__(self, scalar: Union[float]) -> 'Vector':
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError('Cannot divide by zero')
            result_components = []
            for x in self.components:
                result_components.append(x / scalar)
            return Vector(result_components)
        else:
            raise TypeError(f'Unsupported type for division: {type(scalar)}')

    def __pow__(self, other: Union[int, float, 'Vector']) -> Union['Vector', List]:
        if isinstance(other, (int, float)):
            result_components = []
            for x in self.components:
                result_components.append(x ** other)
            return Vector(result_components)
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError('Vectors must have the same dimension for element-wise power')
            result_components = []
            for a, b in zip(self.components, other.components):
                result_components.append(a ** b)
            return Vector(result_components)
        else:
            raise TypeError(f'Unsupported type for power operation: {type(other)}')

    def __rpow__(self, other: Union[int, float]) -> List:
        if isinstance(other, (int, float)):
            result_components = []
            for x in self.components:
                result_components.append(other ** x)
            return result_components
        else:
            raise TypeError(f'Unsupported type for right power operation: {type(other)}')

    def VECTORFLAG():
        pass