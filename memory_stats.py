import torch

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info: %fGB"%(torch.cuda.memory.mem_get_info('cuda:0')[0]/1024/1024/1024))
