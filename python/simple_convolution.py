import cudnn.api_base
import torch
import torch.nn as nn
import numpy as np
import time
import cudnn

INFERENCE_TIMES = 100
kernel_size = 3

im_width = 1920
im_height = 1080

image = torch.rand((im_width, im_height))


a = time.time() * 1000
with torch.no_grad():
    conv = nn.Conv2d(1, 1, kernel_size, padding=1)
torch.cuda.synchronize()
end = time.time() * 1000 - a


res = conv(image)
print(res.shape)

average_inf_time = end / INFERENCE_TIMES

print(f"Average pytorch inference time for CNN (LeNet-5 style, 32x32x3, batch=64): {average_inf_time} ms")


