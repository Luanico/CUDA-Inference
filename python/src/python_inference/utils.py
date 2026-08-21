import torch
import numpy as np
import onnx
import onnx2torch
from onnx2torch import convert
import time


class Benchmarker:
    def __init__(self, model, input_shape, batch_size):
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size

    def inference_average_time(self, inference_times=100):
        """
        Run inference on the model and return the average time per inference in milliseconds.
        """
        input_tensor = torch.randn((self.batch_size, *self.input_shape)).to("cuda")
        _ = self.model(input_tensor)
        a = time.time() * 1000
        with torch.no_grad():
            [self.model(input_tensor) for _ in range(inference_times)]
        torch.cuda.synchronize()
        end = time.time() * 1000 - a
        return end / inference_times
    
def load_ONNX_as_torch(filename):
    modelPytorch = convert(filename)
    return modelPytorch