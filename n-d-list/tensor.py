import json

class Tensor:
    def __init__(self, *args):
        self.data = None
        if len(args) == 0:
            print("0 dim not supported")
            return
        if not all((isinstance(arg, int) and arg > 0 for arg in args)):
            print("dim cant be 0")
            return
        self.data = self.create_tensor(args)

    def create_tensor(self, shape):
        if not shape:
            return 0
        current_dim = shape[0]
        rem_dim = shape[1:]
        return [self.create_tensor(rem_dim) for _ in range(current_dim)]

    def __str__(self):
        return json.dumps(self.data)
    
# 0 dim
a = Tensor()
# 1 dim
a = Tensor(5)
print(a)
# 2 dim
a = Tensor(5, 2)
print(a)
# 3 dim
a = Tensor(2, 5, 2)
print(a)
# error checks
a = Tensor(4, 9, 0)
a = Tensor(1.7)

'''
0 dim not supported
[0, 0, 0, 0, 0]
[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
[[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
dim cant be 0
dim cant be 0
'''
