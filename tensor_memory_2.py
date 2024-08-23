import torch

# difference between tensor_memory_1.py and tensor_memory_2.py is that here the tensor is stored in variables
# and GC frees the tensors at the end of program. Hence you see torch.cuda.memory_allocated piling up.

a = torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float32, device='cuda')

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))
print("------------------------------------------")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float16, device='cuda')

b = print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))
print("------------------------------------------")

#o/p on mi250
'''
torch.cuda.memory_allocated: 16.000244GB
torch.cuda.memory_reserved: 16.001953GB
torch.cuda.max_memory_reserved: 16.001953GB
torch.cuda.memory.mem_get_info free: 47.640625GB
torch.cuda.memory.mem_get_info total: 63.984375GB
------------------------------------------
torch.cuda.memory_allocated: 16.000244GB
torch.cuda.memory_reserved: 24.003906GB
torch.cuda.max_memory_reserved: 24.003906GB
torch.cuda.memory.mem_get_info free: 39.638672GB
torch.cuda.memory.mem_get_info total: 63.984375GB
------------------------------------------
'''
