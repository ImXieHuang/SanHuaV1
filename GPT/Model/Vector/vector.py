import math
from typing import List, Union
import sys
from pathlib import Path

udir = str(Path(__file__).parent.parent)
sys.path.append(udir)

from fraction import fraction

class Vector:
    def __init__(self, components: List[Union[int, float, fraction]]):
        self.components = []
        for x in components:
            if hasattr(x, "FRACTYPE"):
                self.components.append(x)
            else:
                self.components.append(float(x))
        self.dimension = len(components)
        
    def __repr__(self):
        return f"Vector({self.components})"
    
    def __str__(self):
        return str(self.components)
    
    def to_list(self) -> list:
        return self.components.copy()
    
    def magnitude(self) -> float:
        sum_sq = 0.0
        for x in self.components:
            if isinstance(x, fraction):
                sum_sq += x.value ** 2
            else:
                sum_sq += x ** 2
        return math.sqrt(sum_sq)
    
    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        
        normalized_components = []
        for x in self.components:
            if isinstance(x, fraction):
                normalized_components.append(fraction(int(x.value / mag * 1000), 1000))
            else:
                normalized_components.append(x / mag)
        return Vector(normalized_components)
    
    def __add__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        result_components = []
        for a, b in zip(self.components, other.components):
            if isinstance(a, fraction) or isinstance(b, fraction):
                if not isinstance(a, fraction):
                    a = fraction(int(a * 1000), 1000)
                if not isinstance(b, fraction):
                    b = fraction(int(b * 1000), 1000)
                result_components.append(a + b)
            else:
                result_components.append(a + b)
        return Vector(result_components)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        result_components = []
        for a, b in zip(self.components, other.components):
            if isinstance(a, fraction) or isinstance(b, fraction):
                if not isinstance(a, fraction):
                    a = fraction(int(a * 1000), 1000)
                if not isinstance(b, fraction):
                    b = fraction(int(b * 1000), 1000)
                result_components.append(a - b)
            else:
                result_components.append(a - b)
        return Vector(result_components)
    
    def __mul__(self, other: Union[float, 'Vector', fraction]) -> Union[float, int, 'Vector', fraction]:
        if isinstance(other, (int, float, fraction)):
            result_components = []
            for x in self.components:
                if isinstance(x, fraction) or isinstance(other, fraction):
                    if not isinstance(x, fraction):
                        x = fraction(int(x * 1000), 1000)
                    if not isinstance(other, fraction):
                        other_frac = fraction(int(other * 1000), 1000)
                    else:
                        other_frac = other
                    result_components.append(x * other_frac)
                else:
                    result_components.append(x * other)
            return Vector(result_components)
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must have the same dimension")
            
            result = 0.0
            for a, b in zip(self.components, other.components):
                if isinstance(a, fraction) or isinstance(b, fraction):
                    if not isinstance(a, fraction):
                        a = fraction(int(a * 1000), 1000)
                    if not isinstance(b, fraction):
                        b = fraction(int(b * 1000), 1000)
                    result += a.value * b.value
                else:
                    result += a * b
            
            if any(isinstance(x, fraction) for x in self.components) or any(isinstance(x, fraction) for x in other.components):
                return fraction(int(result * 1000), 1000)
            return result
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
    
    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)
    
    def __abs__(self) -> 'Vector':
        return Vector([abs(x) for x in self.components])
    
    def __truediv__(self, scalar: Union[float, fraction]) -> 'Vector':
        if isinstance(scalar, (int, float, fraction)):
            if scalar == 0 or (isinstance(scalar, fraction) and scalar.value == 0):
                raise ValueError("Cannot divide by zero")
            
            result_components = []
            for x in self.components:
                if isinstance(x, fraction) or isinstance(scalar, fraction):
                    if not isinstance(x, fraction):
                        x = fraction(int(x * 1000), 1000)
                    if not isinstance(scalar, fraction):
                        scalar_frac = fraction(int(scalar * 1000), 1000)
                    else:
                        scalar_frac = scalar
                    result_components.append(x / scalar_frac)
                else:
                    result_components.append(x / scalar)
            return Vector(result_components)
        else:
            raise TypeError(f"Unsupported type for division: {type(scalar)}")
    
    def __pow__(self, other: Union[int, float, fraction, 'Vector']) -> Union['Vector', List]:
        if isinstance(other, (int, float, fraction)):
            result_components = []
            for x in self.components:
                if isinstance(x, fraction) or isinstance(other, fraction):
                    if not isinstance(x, fraction):
                        x = fraction(int(x * 1000), 1000)
                    if not isinstance(other, fraction):
                        other_frac = fraction(int(other * 1000), 1000)
                    else:
                        other_frac = other
                    
                    if isinstance(other_frac, fraction) and other_frac.denominator != 1:
                        result_components.append(float(x) ** float(other_frac))
                    else:
                        if other_frac.denominator == 1:
                            exp = int(other_frac.value)
                            if isinstance(x, fraction):
                                result_components.append(x ** exp)
                            else:
                                result_components.append(fraction(x.value ** exp, 1))
                        else:
                            result_components.append(float(x) ** float(other_frac))
                else:
                    result_components.append(x ** other)
            
            if any(isinstance(c, fraction) for c in result_components):
                result_components = [c if isinstance(c, fraction) else 
                                    fraction(int(c * 1000), 1000) for c in result_components]
            
            return Vector(result_components)
        
        elif isinstance(other, Vector):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must have the same dimension for element-wise power")
            
            result_components = []
            for a, b in zip(self.components, other.components):
                if isinstance(a, fraction) or isinstance(b, fraction):
                    if not isinstance(a, fraction):
                        a = fraction(int(a * 1000), 1000)
                    if not isinstance(b, fraction):
                        b_frac = fraction(int(b * 1000), 1000)
                    else:
                        b_frac = b
                    
                    if b_frac.denominator == 1:
                        exp = int(b_frac.value)
                        if isinstance(a, fraction):
                            result_components.append(a ** exp)
                        else:
                            result_components.append(fraction(a.value ** exp, 1))
                    else:
                        result_components.append(float(a) ** float(b_frac))
                else:
                    result_components.append(a ** b)
            
            return Vector(result_components)
        
        else:
            raise TypeError(f"Unsupported type for power operation: {type(other)}")
    
    def __rpow__(self, other: Union[int, float, fraction]) -> List:
        if isinstance(other, (int, float, fraction)):
            result_components = []
            for x in self.components:
                if isinstance(x, fraction) or isinstance(other, fraction):
                    if not isinstance(other, fraction):
                        other_frac = fraction(int(other * 1000), 1000)
                    else:
                        other_frac = other
                    
                    if not isinstance(x, fraction):
                        x = fraction(int(x * 1000), 1000)
                    
                    result_components.append(float(other_frac) ** float(x))
                else:
                    result_components.append(other ** x)
            
            return result_components
        else:
            raise TypeError(f"Unsupported type for right power operation: {type(other)}")
    
    def VECTORFLAG(): pass