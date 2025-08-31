def maxpool1d(lst, pool_size, no_pad=True, stride=None):
    result = []
    if not no_pad and len(lst) % pool_size != 0:
        pad = len(lst) % pool_size
        for _ in range(pad):
            lst.append(0)
    print("input with pad ", lst)
    if stride is None:
        stride = pool_size
    for i in range(0, len(lst)-pool_size+1, stride):
        window = lst[i:i+pool_size]
        maxi = max(window)
        result.append(maxi)
    return result


lst = [2, 4, 1, 5, 3, 8, 7, 6]
result = maxpool1d(lst, pool_size=3, no_pad = True, stride=2)
print(result)  # Output: [4, 5, 8]
lst = [2, 4, 1, 5, 3, 8, 7, 6]
result = maxpool1d(lst, pool_size=3, no_pad = True)
print(result)  # Output: [4, 8]

lst = [2, 4, 1, 5, 3, 8, 7, 6]
result = maxpool1d(lst, pool_size=3, no_pad = False, stride=2)
print(result)  # Output: [4, 5, 8, 7]
lst = [2, 4, 1, 5, 3, 8, 7, 6]
result = maxpool1d(lst, pool_size=3, no_pad = False)
print(result)  # Output: [4, 8, 7]
