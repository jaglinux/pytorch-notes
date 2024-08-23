import torch


torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float32, device='cuda')
# 16GB is allocated for above tensor, formula below
# ((int(2 ** 16) * int(2 ** 16) + 1) * 4 ) // (1024*1024*1024)

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))
print("------------------------------------------")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.randn(int(2 ** 16), int(2 ** 16) + 1, dtype=torch.float16, device='cuda')
# 8GB is allocated for above tensor, formula below
# ((int(2 ** 16) * int(2 ** 16) + 1) * 2 ) // (1024*1024*1024)

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))
print("------------------------------------------")

# o/p on mi250
'''
torch.cuda.memory_allocated: 0.000000GB
torch.cuda.memory_reserved: 16.001953GB
torch.cuda.max_memory_reserved: 16.001953GB
torch.cuda.memory.mem_get_info free: 47.640625GB
torch.cuda.memory.mem_get_info total: 63.984375GB
------------------------------------------
torch.cuda.memory_allocated: 0.000000GB
torch.cuda.memory_reserved: 8.001953GB
torch.cuda.max_memory_reserved: 8.001953GB
torch.cuda.memory.mem_get_info free: 55.640625GB
torch.cuda.memory.mem_get_info total: 63.984375GB
------------------------------------------

'''
