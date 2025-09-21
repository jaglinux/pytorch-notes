import torch

def matmul(a, b):
    #print("matmul a list is ", a)
    #print("matmul b list is ", b)
    assert len(a[0]) == len(b), "k dimension should be same"
    
    result = [ [0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                #print(i, j, k)
                result[i][j] += (a[i][k]*b[k][j])
    return torch.FloatTensor(result)

M, N, K = 4, 2, 4
a = torch.randn(M, K)
b = torch.randn(N, K)
result_pt = a @ b.t()
print("input a is ", a)
print("input b is ", b.t())
print("shapes are ", a.shape, b.t().shape, result_pt.shape)
print("result through torch is ", result_pt)
result_matmul = matmul(a, b.t())
print("result through manual matmul is ", result_matmul, type(result_matmul))
print("compare both the tensors ", torch.isclose(result_pt, result_matmul).all())

'''
input a is  tensor([[-0.3262, -0.6583,  0.6583,  0.4244],
        [-0.1956, -1.7367, -0.9480, -0.5192],
        [ 1.1214, -0.3556,  1.7449, -1.9608],
        [-0.7172,  0.1358,  0.1972,  1.2248]])
input b is  tensor([[ 0.8475, -0.7289],
        [-0.1650, -0.6970],
        [ 0.3761,  0.4684],
        [ 0.0164, -0.5301]])
shapes are  torch.Size([4, 4]) torch.Size([4, 2]) torch.Size([4, 2])
result through torch is  tensor([[ 0.0867,  0.7800],
        [-0.2442,  1.1842],
        [ 1.6332,  1.2871],
        [-0.5361, -0.1288]])
result through manual matmul is  tensor([[ 0.0867,  0.7800],
        [-0.2442,  1.1842],
        [ 1.6332,  1.2871],
        [-0.5361, -0.1288]]) <class 'torch.Tensor'>
compare both the tensors  tensor(True)
'''
