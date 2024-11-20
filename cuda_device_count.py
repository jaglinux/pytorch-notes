# 8 gpu node and set HIP_VISIBLE_DEVICES=0,1

import torch

# prints 2
torch.cuda.device_count()
#prints [0, 1] (gpu indices)
torch.cuda._parse_visible_devices()

# this prints the actual GPUs available.
#prints 8
torch.cuda._raw_device_count_amdsmi()
