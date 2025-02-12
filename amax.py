import torch

x = torch.tensor([[1, 5, 3], [4, 2, 6]], dtype=torch.float32)

# torch.max
max_values_single = torch.max(x)
max_values_dim0 = torch.max(x, dim=0)
max_values_dim1 = torch.max(x, dim=1)

print("torch.max (single tensor):", max_values_single)
print("torch.max (dim=0):", max_values_dim0)
print("torch.max (dim=1):", max_values_dim1)

print("----------------------------------------")
# torch.amax
max_values_amax_single = torch.amax(x)
max_values_amax_dim0 = torch.amax(x, dim=0)
max_values_amax_dim1 = torch.amax(x, dim=1)
max_values_amax_multi_dim = torch.amax(x, dim=(0, 1))

print("torch.amax (single tensor):", max_values_amax_single)
print("torch.amax (dim=0):", max_values_amax_dim0)
print("torch.amax (dim=1):", max_values_amax_dim1)
print("torch.amax (dim=(0,1)):", max_values_amax_multi_dim)

'''
torch.max (single tensor): tensor(6.)
torch.max (dim=0): torch.return_types.max(
values=tensor([4., 5., 6.]),
indices=tensor([1, 0, 1]))
torch.max (dim=1): torch.return_types.max(
values=tensor([5., 6.]),
indices=tensor([1, 2]))
----------------------------------------
torch.amax (single tensor): tensor(6.)
torch.amax (dim=0): tensor([4., 5., 6.])
torch.amax (dim=1): tensor([5., 6.])
torch.amax (dim=(0,1)): tensor(6.)

'''
