import math
import random

e = math.e

def softmax(t1):
    total = 0
    t1_e = []
    for t in t1:
        a = e ** t
        t1_e.append(a)
        total += a

    t1 = [t/total for t in t1_e]

    print(t1)
    print(sum(t1))

    index = 0
    maxi = 0

    for i,value in enumerate(t1):
        if i == 0:
            maxi = value
        else:
            if value > maxi:
                maxi = value
                index = i

    return index, t1

embedding = 10
t1 = [random.randint(-10, 10) for _ in range(embedding)]
print(t1)
index , t = softmax(t1)
print(index, t)
        

'''
[7, 2, 2, -6, -10, 8, -10, 0, 1, 3]
[0.2664203991152982, 0.0017951265287140762, 0.0017951265287140762, 6.021978627399381e-07, 1.102963859351222e-08, 0.7242057296459212, 1.102963859351222e-08, 0.00024294395720907686, 0.0006603901442153654, 0.0048796598227882374]
1.0
5 [0.2664203991152982, 0.0017951265287140762, 0.0017951265287140762, 6.021978627399381e-07, 1.102963859351222e-08, 0.7242057296459212, 1.102963859351222e-08, 0.00024294395720907686, 0.0006603901442153654, 0.0048796598227882374]
'''
