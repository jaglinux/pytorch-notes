import torch

a = torch.zeros(4096, dtype=torch.float16, device='cuda')
print(a, a.dtype, a.shape, a.dim())
a.fill_(16.0)
print(a, a.dtype, a.shape, a.dim())
b = a.sum()
print(b, b.dtype, b.shape, b.dim())

a = torch.zeros(4096, dtype=torch.float32, device='cuda')
print(a, a.dtype, a.shape, a.dim())
a.fill_(16.0)
print(a, a.dtype, a.shape, a.dim())
b = a.sum()
print(b, b.dtype, b.shape, b.dim())

'''

tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0', dtype=torch.float16) torch.float16 torch.Size([4096]) 1
tensor([16., 16., 16.,  ..., 16., 16., 16.], device='cuda:0',
       dtype=torch.float16) torch.float16 torch.Size([4096]) 1
tensor(inf, device='cuda:0', dtype=torch.float16) torch.float16 torch.Size([]) 0
tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0') torch.float32 torch.Size([4096]) 1
tensor([16., 16., 16.,  ..., 16., 16., 16.], device='cuda:0') torch.float32 torch.Size([4096]) 1
tensor(65536., device='cuda:0') torch.float32 torch.Size([]) 0

'''
