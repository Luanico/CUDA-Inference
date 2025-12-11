import torch
import numpy as np
import onnx
import time

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(6, 16, 5, padding=2)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.relu2 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120, 84)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x




def saveRandomModel(filename):
    modelPytorch = CNN()
    dummy_input = torch.randn((64, 3, 32, 32))
    torch.onnx.export(modelPytorch, dummy_input, filename, input_names=['input'], output_names=['output'])
    
    model = onnx.load(filename)
    from onnx.external_data_helper import load_external_data_for_model
    load_external_data_for_model(model, '.')
    onnx.save(model, filename, save_as_external_data=False)

def loadONNXW(filename):
    model = onnx.load(filename)

    modelPytorch = CNN()
    modelPytorch.conv1.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[0]))
    modelPytorch.conv1.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[1]))

    modelPytorch.conv2.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[2]))
    modelPytorch.conv2.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[3]))

    modelPytorch.fc1.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[4]))
    modelPytorch.fc1.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[5]))

    modelPytorch.fc2.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[6]))
    modelPytorch.fc2.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[7]))

    modelPytorch.fc3.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[8]))
    modelPytorch.fc3.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[9]))

    return modelPytorch


def inference(model):
    input_tensor = torch.randn((64, 3, 32, 32)).to("cuda")
    _ = model(input_tensor)
    INFERENCE_TIMES = 100
    a = time.time() * 1000
    with torch.no_grad():
        [model(input_tensor) for _ in range(INFERENCE_TIMES)]
    torch.cuda.synchronize()
    end = time.time() * 1000 - a
    return end / INFERENCE_TIMES


FILENAME = "cnn_fixed.onnx"

saveRandomModel(FILENAME)
model = loadONNXW(FILENAME).to("cuda")
model.eval()
average_inf_time = inference(model)

print(f"Average pytorch inference time for CNN (LeNet-5 style, 32x32x3, batch=64): {average_inf_time} ms")