class kernel:
    def __init__(self, weights: list, size: int):
        self.weights = weights
        self.size = size

        i = len(weights)
        dim = 0
        while i > 1:
            i /= size 
            dim += 1

def conv(x: list):
    pass

def pool(x: list):
    pass

if __name__ == "__main__":
    kernel([1,1,1,1,1,1,1], 3)