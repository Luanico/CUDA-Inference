import torch
import numpy as np
import onnx
import time

class CNN(torch.nn.Module):
    def __init__(self, in_channel, out_channel, input_size, output_size=10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel * 2, 3, padding=1)
        self.relu2 = torch.nn.ReLU()
        
        pooled_size = input_size // 2
        self.fc1 = torch.nn.Linear(out_channel * 2 * pooled_size * pooled_size, 512)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 256)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x




def saveRandomModel(filename, input_size, in_channel, out_channel, batch_size):
    modelPytorch = CNN(in_channel, out_channel, input_size)
    dummy_input = torch.randn((batch_size, in_channel, input_size, input_size))
    torch.onnx.export(modelPytorch, dummy_input, filename, input_names=['input'], output_names=['output'])
    
    model = onnx.load(filename)
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    load_external_data_for_model(model, '.')
    onnx.save(model, filename, save_as_external_data=False)

def loadONNXW(filename, input_size, in_channel, out_channel):
    model = onnx.load(filename)

    modelPytorch = CNN(in_channel, out_channel, input_size)
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




def inference(model, INPUT_SIZE, batch_size, in_channel):
    input_tensor = torch.randn((batch_size, in_channel, INPUT_SIZE, INPUT_SIZE)).to("cuda")
    _ = model(input_tensor)
    INFERENCE_TIMES = 100
    a = time.time() * 1000
    with torch.no_grad():
        [model(input_tensor) for _ in range(INFERENCE_TIMES)]
    torch.cuda.synchronize()
    end = time.time() * 1000 - a
    return end / INFERENCE_TIMES


architectures = {
    (32, 3, 16, 64) : "cnn_small.onnx",
    (64, 3, 32, 64) : "cnn_medium.onnx",
    (128, 3, 64, 64) : "cnn_large.onnx",
    (32, 3, 6, 64) : "cnn_fixed.onnx",
}

for key, val in architectures.items():
    saveRandomModel(val, key[0], key[1], key[2], key[3])
    model = loadONNXW(val, key[0], key[1], key[2]).to("cuda")
    model.eval()
    average_inf_time = inference(model, key[0], key[3], key[1])

    print(f"Average pytorch inference time for CNN with architecure (input_size, in_channel, out_channel, batch) : {key} = {average_inf_time} ms")