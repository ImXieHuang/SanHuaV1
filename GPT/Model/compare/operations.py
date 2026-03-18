def distance(a, b, visited=None, depth=0, max_depth=10):
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return 1.0
    
    key = (id(a), id(b))
    if key in visited:
        return 0.0
    visited.add(key)
    
    if a is b:
        return 0.0
    
    if a is None or b is None:
        return 1.0 if a is not b else 0.0
    
    if type(a) != type(b):
        try:
            fa, fb = float(a), float(b)
            base = max(abs(fa), abs(fb), 1.0)
            return min(abs(fa - fb) / base, 1.0)
        except:
            return 1.0
    
    if isinstance(a, (int, float, complex)):
        if a == b:
            return 0.0
        fa, fb = float(a), float(b)
        base = max(abs(fa), abs(fb), 1.0)
        return min(abs(fa - fb) / base, 1.0)
    
    if isinstance(a, str):
        if a == b:
            return 0.0
        len_a, len_b = len(a), len(b)
        diff = 0
        for i in range(min(len_a, len_b)):
            if a[i] != b[i]:
                diff += 1
        diff += abs(len_a - len_b)
        return min(diff / max(len_a, len_b, 1), 1.0)
    
    if isinstance(a, bool):
        return 0.0 if a == b else 1.0
    
    if hasattr(a, '__str__') and hasattr(b, '__str__'):
        try:
            str_a, str_b = str(a), str(b)
            if str_a != str_b:
                return distance(str_a, str_b, visited, depth+1, max_depth)
            elif str_a == str_b:
                return 0.0
        except:
            pass
    
    if hasattr(a, '__list__') and hasattr(b, '__list__'):
        try:
            list_a, list_b = a.__list__(), b.__list__()
            return distance(list_a, list_b, visited, depth+1, max_depth)
        except:
            pass
    
    if isinstance(a, (list, tuple, set)):
        if not isinstance(b, (list, tuple, set)):
            return 1.0
        
        len_a, len_b = len(a), len(b)
        if len_a != len_b and not isinstance(a, set):
            diff = abs(len_a - len_b)
        else:
            diff = 0
            
        if isinstance(a, set):
            if isinstance(b, set):
                union_len = len(a | b)
                if union_len == 0:
                    return 0.0
                inter_len = len(a & b)
                diff = 1.0 - (inter_len / union_len)
            else:
                return 1.0
        else:
            paired_a = list(a)
            paired_b = list(b)
            min_len = min(len_a, len_b)
            
            for i in range(min_len):
                d = distance(paired_a[i], paired_b[i], visited, depth+1, max_depth)
                diff += d
            
            if len_a != len_b:
                diff += abs(len_a - len_b)
            
            total = max(len_a, len_b)
            if total == 0:
                return 0.0
            diff = diff / total
        
        return min(diff, 1.0)
    
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return 1.0
        
        keys_a, keys_b = set(a.keys()), set(b.keys())
        all_keys = keys_a | keys_b
        
        if not all_keys:
            return 0.0
        
        diff = 0
        for key in all_keys:
            if key in a and key in b:
                d = distance(a[key], b[key], visited, depth+1, max_depth)
                diff += d
            else:
                diff += 1
        
        return min(diff / len(all_keys), 1.0)
    
    try:
        if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
            return distance(a.__dict__, b.__dict__, visited, depth+1, max_depth)
    except:
        pass
    
    try:
        if hasattr(a, '__slots__') and hasattr(b, '__slots__'):
            dict_a = {slot: getattr(a, slot) for slot in a.__slots__ if hasattr(a, slot)}
            dict_b = {slot: getattr(b, slot) for slot in b.__slots__ if hasattr(b, slot)}
            return distance(dict_a, dict_b, visited, depth+1, max_depth)
    except:
        pass
    
    try:
        if hasattr(a, '__iter__') and hasattr(b, '__iter__'):
            try:
                items_a = list(a)
                items_b = list(b)
                return distance(items_a, items_b, visited, depth+1, max_depth)
            except:
                pass
    except:
        pass
    
    try:
        return 0.0 if a == b else 1.0
    except:
        return 1.0