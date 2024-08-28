import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

#o/p on GPU
'''
True
2
0
AMD Instinct MI210
'''
