import random
import torch

def _isList(a):
    return isinstance(a, list) and len(a) > 0 and all(isinstance(i, list) for i in a)

def _isMatrix(a):
    col = len(a[0])
    return all(len(i) == col for i in a)

def matmul(a, b):
    assert _isList(a) and _isList(b), "_isList failed"
    assert _isMatrix(a) and _isMatrix(b), "isMatrix failed"
    M , K1 = len(a), len(a[0])
    K2, N = len(b), len(b[0])
    assert K1 == K2, "K is not equal"
    K = K1
    print(f"M is {M} K is {K} N is {N}")
    print(f"total elements in A is {M*K}")
    print(f"total elements in B is {K*N}")
    c = [[0 for _ in range(N)]for _ in range(M)]
    print(f"c row is {len(c)} and c col is {len(c[0])} and total elements are {len(c)*len(c[0])}")
    for i in range(M):
        for j in range(N):
            acc = 0
            for k in range(K):
                acc += a[i][k]*b[k][j]
            c[i][j] = acc
    print("matmul function end")
    return c

def compare(c, c_torch):
    for i in range(len(c)):
        for j in range(len(c[0])):
            if c[i][j] != c_torch[i][j]:
                print("compare failed !!!")
    print("compare success !!!")

if __name__ == "__main__":
    M , K , N = 24, 12, 15
    a = [[random.randint(1,9) for _ in range(K)] for _ in range(M)]
    b = [[random.randint(1,9) for _ in range(N)] for _ in range(K)]
    print(f"a is {a}")
    print(f"b is {b}")
    print(f"len of a is {len(a)}")
    print(f"len of a[0] is {len(a[0])}")
    print(f"len of b is {len(b)}")
    print(f"len of b[0] is {len(b[0])}")
    c = matmul(a, b)
    c_torch = torch.tensor(a) @ torch.tensor(b)
    print(f"c is {c}")
    print(f"c_torch is {c_torch}")
    compare(c, c_torch.tolist())

