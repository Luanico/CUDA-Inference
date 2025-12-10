import torch
import numpy as np
import onnx
import time

class CNN(torch.nn.Module):
    def __init__(self, in_channel, out_channel, input_size, output_size=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(out_channel * input_size * input_size, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
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
    modelPytorch.conv.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[0]))
    modelPytorch.conv.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[1]))

    modelPytorch.output.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[2]))
    modelPytorch.output.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[3]))

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
    (32, 16, 32, 64) : "cnn_small.onnx",
    (64, 32, 128, 128) : "cnn_medium.onnx",
    (128, 64, 256, 256) : "cnn_large.onnx",
    (224, 64, 512, 512) : "cnn_huge.onnx",
}

for key, val in architectures.items():
    saveRandomModel(val, key[0], key[1], key[2], key[3])
    model = loadONNXW(val, key[0], key[1], key[2]).to("cuda")
    model.eval()
    average_inf_time = inference(model, key[0], key[3], key[1])

    print(f"Average pytorch inference time for CNN with architecure (input, hidden, output, batch) : {key} = {average_inf_time} ms")