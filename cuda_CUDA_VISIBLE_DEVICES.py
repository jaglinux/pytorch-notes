import torch

a = torch.cuda.device_count()
print(a)
torch.cuda.set_device(1)
print("current device ", torch.cuda.current_device())

#i/p
# CUDA_VISIBLE_DEVICES=0,1 python cuda_CUDA_VISIBLE_DEVICES.py
# o/p
'''
2
current device  1
'''

#i/p
# CUDA_VISIBLE_DEVICES=0 python cuda_CUDA_VISIBLE_DEVICES.py
# o/p
'''
1
File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/cuda/__init__.py", line 399, in set_device
    torch._C._cuda_setDevice(device)
RuntimeError: HIP error: invalid device ordinal
'''

#i/p
# python cuda_CUDA_VISIBLE_DEVICES.py
# o/p
'''
8
current device  1
'''
