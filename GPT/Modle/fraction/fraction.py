class fraction:
    def __init__(self, numerator: int, denominator: int):
        self.numerator = numerator
        self.denominator = denominator
        self.value = self.numerator / self.denominator
        
        def verify():
            if not isinstance(self.numerator, int):
                raise TypeError("Numerator must be an integer.")
            if not isinstance(self.denominator, int):
                raise TypeError("Denominator must be an integer.")
            if self.denominator == 0:
                raise ValueError("Denominator cannot be zero.")
        
        verify()
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        if not isinstance(other, fraction):
            raise TypeError("Operand must be a fraction or number.")
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return fraction(new_numerator, new_denominator)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        if not isinstance(other, fraction):
            raise TypeError("Operand must be a fraction or number.")
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return fraction(new_numerator, new_denominator)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        return other - self
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        if not isinstance(other, fraction):
            raise TypeError("Operand must be a fraction or number.")
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return fraction(new_numerator, new_denominator)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        if not isinstance(other, fraction):
            raise TypeError("Operand must be a fraction or number.")
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return fraction(new_numerator, new_denominator)
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = fraction(int(other * 1000), 1000)
        return other / self
    
    def __pow__(self, power):
        if isinstance(power, (int, float)):
            result = self.value ** power
            return fraction(int(result * 1000), 1000)
        elif isinstance(power, fraction):
            result = self.value ** power.value
            return fraction(int(result * 1000), 1000)
        raise TypeError("Power must be a number.")
    
    def __rpow__(self, base):
        if isinstance(base, (int, float)):
            result = base ** self.value
            return fraction(int(result * 1000), 1000)
        raise TypeError("Base must be a number.")
    
    def __lt__(self, other):
        if isinstance(other, fraction):
            return self.value < other.value
        elif isinstance(other, (int, float)):
            return self.value < other
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, fraction):
            return self.value <= other.value
        elif isinstance(other, (int, float)):
            return self.value <= other
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, fraction):
            return self.value > other.value
        elif isinstance(other, (int, float)):
            return self.value > other
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, fraction):
            return self.value >= other.value
        elif isinstance(other, (int, float)):
            return self.value >= other
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, fraction):
            return self.numerator * other.denominator == other.numerator * self.denominator
        elif isinstance(other, (int, float)):
            return abs(self.value - other) < 1e-10
        return NotImplemented
    
    def __neg__(self):
        return fraction(-self.numerator, self.denominator)
    
    def __abs__(self):
        return fraction(abs(self.numerator), abs(self.denominator))
    
    def riverse(self):
        return fraction(self.denominator, self.numerator)
    
    def __str__(self):
        return f"{self.numerator}/{self.denominator}"
    
    def __repr__(self):
        return f"fraction({self.numerator}, {self.denominator})"