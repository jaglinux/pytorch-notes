import torch
import torch.nn.functional as F

# Example of target with class indices
input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(2, (3, ), dtype=torch.int64)
# default entropy reduction is "mean"
# 'mean': the sum of the output will be divided by the number of elements in the output
loss = F.cross_entropy(input, target)
print("input is ", input)
print("input grad is ", input.grad)
print("target is ", target)
print("loss is ", loss)

print("---------------------------")

loss.backward()
print("input grad after backward is ", input.grad)
input.grad.data.zero_()
print("just as an experiment, make gradient zero ")
print("input grad after backward is ", input.grad)

#output
'''
input is  tensor([[-0.7657,  0.3868,  0.4532, -0.8396, -0.6728],
        [ 0.6048,  0.4591, -3.0205,  0.8796, -0.2060],
        [ 0.7906, -0.2274,  1.5982, -0.2628,  0.9909]], requires_grad=True)
input grad is  None
target is  tensor([0, 0, 1])
loss is  tensor(2.0721, grad_fn=<NllLossBackward0>)
---------------------------
input grad after backward is  tensor([[-0.2985,  0.1102,  0.1178,  0.0323,  0.0382],
        [-0.2421,  0.0789,  0.0024,  0.1201,  0.0406],
        [ 0.0644, -0.3101,  0.1445,  0.0225,  0.0787]])
just as an experiment, make gradient zero
input grad after backward is  tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])

'''
