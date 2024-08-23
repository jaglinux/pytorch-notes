import torch

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
print("torch.cuda.memory.mem_get_info free: %fGB"%(torch.cuda.memory.mem_get_info()[0]/1024/1024/1024))
print("torch.cuda.memory.mem_get_info total: %fGB"%(torch.cuda.memory.mem_get_info()[1]/1024/1024/1024))

# o/p on mi250 without any load
'''
torch.cuda.memory_allocated: 0.000000GB
torch.cuda.memory_reserved: 0.000000GB
torch.cuda.max_memory_reserved: 0.000000GB
torch.cuda.memory.mem_get_info free: 63.787109GB
torch.cuda.memory.mem_get_info total: 63.984375GB
'''

# definitions
'''
torch.cuda.memory_allocated
Return the current GPU memory occupied by tensors in bytes for a given device.

torch.cuda.memory_reserved
Return the current GPU memory managed by the caching allocator in bytes for a given device.

torch.cuda.max_memory_reserved
Return the maximum GPU memory managed by the caching allocator in bytes for a given device.
By default, this returns the peak cached memory since the beginning of this program. reset_peak_memory_stats() can be used to reset the starting point in tracking this metric.

torch.cuda.mem_get_info
Return the global free and total GPU memory for a given device using cudaMemGetInfo. Return tuple (free mem, total mem)
This is not specific to this program.
Used memory includes memory allocated in this program + other PT instances + any other program using GPU memory.
Total mem is GPU DRAM (HBM, global). for mi250 it should be 64GB, for mi300 it should be 192GB

'''
print("end")

