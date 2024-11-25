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
print(torch.cuda.get_device_properties(0))
'''
_CudaDeviceProperties(name='AMD Instinct MI250X/MI250', major=9, minor=0, gcnArchName='gfx90a', 
total_memory=65520MB, multi_processor_count=110, L2_cache_size=8MB)
'''
