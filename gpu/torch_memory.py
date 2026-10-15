import torch
a = torch.empty(1024, 1024, device="cuda", dtype=torch.float16)
print(torch.cuda.is_available())
free, total = torch.cuda.mem_get_info(0)
print(f"Free {free/(1024**3)} Total {total/(1024**3)} Allocated {(total - free)/(1024**3)} GiB")
print(f"Allocated {torch.cuda.memory_allocated(0) // (1024*1024)} MiB")
print(f"Reserved {torch.cuda.memory_reserved(0) // (1024*1024)} MiB")
print(f"Max allocated {torch.cuda.max_memory_allocated(0) // (1024*1024)} MiB")
print(f"Max reserved {torch.cuda.max_memory_reserved(0) // (1024*1024)} MiB")


'''
True
Free 251.36328125 Total 251.984375 Allocated 0.62109375 GiB
Allocated 2 MiB
Reserved 20 MiB
Max allocated 2 MiB
Max reserved 20 MiB
'''
