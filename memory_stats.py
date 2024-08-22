import torch

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))

# o/p on mi250 without any load
torch.cuda.memory_allocated: 0.000000GB
torch.cuda.memory_reserved: 0.000000GB
torch.cuda.max_memory_reserved: 0.000000GB
torch.cuda.memory.mem_get_info free: 63.787109GB
torch.cuda.memory.mem_get_info total: 63.984375GB
