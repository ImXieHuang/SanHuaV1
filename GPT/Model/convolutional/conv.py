class kernel:
    def __init__(self, weights: list, size: int):
        self.weights = weights
        self.size = size

        i = len(weights)
        dim = 0
        while i > 1:
            i /= size 
            dim += 1
        self.dim = dim
        self.shape = tuple([size] * dim)

def conv(x: list, kernel_obj: kernel, stride: int = 1):
    input_shape = []
    temp = x
    while isinstance(temp, list):
        input_shape.append(len(temp))
        if len(temp) > 0:
            temp = temp[0]
        else:
            break
    
    dim = len(input_shape)
    
    if dim != kernel_obj.dim:
        raise ValueError(f"Input dimension {dim} doesn't match kernel dimension {kernel_obj.dim}")
    
    kernel_shape = kernel_obj.shape
    kernel_size = kernel_obj.size
    
    output_shape = []
    for i in range(dim):
        output_size = (input_shape[i] - kernel_shape[i]) // stride + 1
        if output_size <= 0:
            raise ValueError(f"Invalid dimensions")
        output_shape.append(output_size)
    
    def build_output(shape):
        if len(shape) == 1:
            return [0] * shape[0]
        return [build_output(shape[1:]) for _ in range(shape[0])]
    
    output = build_output(output_shape)
    
    weights_nd = kernel_obj.weights
    for _ in range(dim - 1):
        new_weights = []
        for i in range(0, len(weights_nd), kernel_size):
            new_weights.append(weights_nd[i:i+kernel_size])
        weights_nd = new_weights
    
    output_indices = []
    def gen_indices(current, shape, result):
        if len(current) == len(shape):
            result.append(tuple(current))
            return
        for i in range(shape[len(current)]):
            current.append(i)
            gen_indices(current, shape, result)
            current.pop()
    
    all_output_pos = []
    gen_indices([], output_shape, all_output_pos)
    
    all_weight_pos = []
    gen_indices([], kernel_shape, all_weight_pos)
    
    for out_pos in all_output_pos:
        start_pos = [out_pos[i] * stride for i in range(dim)]
        
        conv_val = 0
        for w_pos in all_weight_pos:
            input_pos = [start_pos[i] + w_pos[i] for i in range(dim)]
            
            temp = x
            for idx in input_pos:
                temp = temp[idx]
            input_val = temp
            
            temp_w = weights_nd
            for idx in w_pos:
                temp_w = temp_w[idx]
            weight_val = temp_w
            
            conv_val += input_val * weight_val
        
        temp = output
        for idx in out_pos[:-1]:
            temp = temp[idx]
        temp[out_pos[-1]] = conv_val
    
    return output

def pool(x: list, pool_size: int = 2, stride: int = 2, pool_type: str = 'max'):
    input_shape = []
    temp = x
    while isinstance(temp, list):
        input_shape.append(len(temp))
        if len(temp) > 0:
            temp = temp[0]
        else:
            break
    
    dim = len(input_shape)
    
    output_shape = []
    for i in range(dim):
        output_size = (input_shape[i] - pool_size) // stride + 1
        if output_size <= 0:
            raise ValueError(f"Invalid dimensions")
        output_shape.append(output_size)
    
    def build_output(shape):
        if len(shape) == 1:
            return [0] * shape[0]
        return [build_output(shape[1:]) for _ in range(shape[0])]
    
    output = build_output(output_shape)
    
    output_indices = []
    def gen_indices(current, shape, result):
        if len(current) == len(shape):
            result.append(tuple(current))
            return
        for i in range(shape[len(current)]):
            current.append(i)
            gen_indices(current, shape, result)
            current.pop()
    
    all_output_pos = []
    gen_indices([], output_shape, all_output_pos)
    
    for out_pos in all_output_pos:
        start_pos = [out_pos[i] * stride for i in range(dim)]
        
        window_values = []
        
        window_indices = []
        def gen_window_indices(current, start, size, result):
            if len(current) == dim:
                result.append(tuple(current))
                return
            d = len(current)
            for i in range(start[d], start[d] + size):
                current.append(i)
                gen_window_indices(current, start, size, result)
                current.pop()
        
        gen_window_indices([], start_pos, pool_size, window_indices)
        
        for idx in window_indices:
            temp = x
            for i in idx:
                temp = temp[i]
            window_values.append(temp)
        
        if pool_type == 'max':
            pool_val = max(window_values)
        elif pool_type == 'avg':
            pool_val = sum(window_values) / len(window_values)
        elif pool_type == 'min':
            pool_val = min(window_values)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        
        temp = output
        for idx in out_pos[:-1]:
            temp = temp[idx]
        temp[out_pos[-1]] = pool_val
    
    return output

if __name__ == "__main__":
    k1 = kernel([1, 2, 1], 3)
    input_1d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"Input 1D: {input_1d}")
    conv_1d = conv(input_1d, k1, stride=1)
    print(f"Conv 1D result: {conv_1d}")
    pool_1d = pool(conv_1d, pool_size=2, stride=2, pool_type='max')
    print(f"Pool 1D result: {pool_1d}")
    
    k2 = kernel([1, 0, -1, 1, 0, -1, 1, 0, -1], 3)
    input_2d = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]
    print("Input 2D:")
    for row in input_2d:
        print(row)
    conv_2d = conv(input_2d, k2, stride=1)
    print("\nConv 2D result:")
    for row in conv_2d:
        print(row)
    pool_2d = pool(conv_2d, pool_size=2, stride=2, pool_type='max')
    print("\nPool 2D result:")
    for row in pool_2d:
        print(row)
    
    k3 = kernel([1] * 27, 3)
    input_3d = [[[i+j+k for k in range(4)] for j in range(4)] for i in range(4)]
    print("Input 3D shape: 4x4x4")
    conv_3d = conv(input_3d, k3, stride=1)
    print(f"Conv 3D result shape: {len(conv_3d)}x{len(conv_3d[0])}x{len(conv_3d[0][0])}")
    pool_3d = pool(conv_3d, pool_size=2, stride=2, pool_type='avg')
    print(f"Pool 3D result shape: {len(pool_3d)}x{len(pool_3d[0])}x{len(pool_3d[0][0])}")
    
    k4 = kernel([1] * 16, 2)
    input_4d = [[[[i+j+k+l for l in range(4)] for k in range(4)] for j in range(4)] for i in range(4)]
    print("Input 4D shape: 4x4x4x4")
    conv_4d = conv(input_4d, k4, stride=1)
    print(f"Conv 4D result shape: {len(conv_4d)}x{len(conv_4d[0])}x{len(conv_4d[0][0])}x{len(conv_4d[0][0][0])}")
    pool_4d = pool(conv_4d, pool_size=2, stride=2, pool_type='max')
    print(f"Pool 4D result shape: {len(pool_4d)}x{len(pool_4d[0])}x{len(pool_4d[0][0])}x{len(pool_4d[0][0][0])}")
    
    test_data = [1, 5, 3, 8, 2, 7, 4, 6]
    print(f"Test data: {test_data}")
    print(f"Max pool: {pool(test_data, pool_size=2, stride=2, pool_type='max')}")
    print(f"Avg pool: {pool(test_data, pool_size=2, stride=2, pool_type='avg')}")
    print(f"Min pool: {pool(test_data, pool_size=2, stride=2, pool_type='min')}")