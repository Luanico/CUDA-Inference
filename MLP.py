import torch
import numpy as np
import onnx
import time

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x)
        return x


INPUT_SIZE = 2048
HIDDEN_SIZE = 4096
OUTPUT_SIZE = 10
batch_size=1024



def saveRandomModel(filename):
    modelPytorch = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    dummy_input = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(modelPytorch, dummy_input, filename, input_names=['input'], output_names=['output'])

def loadONNXW(filename):
    model = onnx.load(filename)

    modelPytorch = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    modelPytorch.linear.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[0]))
    modelPytorch.linear.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[1]))

    modelPytorch.output.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[2]))
    modelPytorch.output.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[3]))

    return modelPytorch


FILENAME = "mlp_2048_4096_10.onnx"

#saveRandomModel(FILENAME)

model = loadONNXW(FILENAME).to("cuda")
model.eval()

def inference(model):
    a = time.time() * 1000
    input_tensor = torch.randn((batch_size, INPUT_SIZE)).to("cuda")
    INFERENCE_TIMES = 100
    with torch.no_grad():
        [model(input_tensor) for _ in range(INFERENCE_TIMES)]
    torch.cuda.synchronize()
    end = time.time() * 1000 - a
    return end / INFERENCE_TIMES

print(f"Average pytorch inference time : {inference(model)}")