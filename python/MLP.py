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




def saveRandomModel(filename, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
    modelPytorch = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    dummy_input = torch.randn(1, INPUT_SIZE)
    torch.onnx.export(modelPytorch, dummy_input, filename, input_names=['input'], output_names=['output'])
    
    model = onnx.load(filename)
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    load_external_data_for_model(model, '.')
    onnx.save(model, filename, save_as_external_data=False)

def loadONNXW(filename, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
    model = onnx.load(filename)

    modelPytorch = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    modelPytorch.linear.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[0]))
    modelPytorch.linear.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[1]))

    modelPytorch.output.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[2]))
    modelPytorch.output.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(model.graph.initializer[3]))

    return modelPytorch




def inference(model, INPUT_SIZE, batch_size):
    input_tensor = torch.randn((batch_size, INPUT_SIZE)).to("cuda")
    _ = model(input_tensor)
    INFERENCE_TIMES = 100
    a = time.time() * 1000
    with torch.no_grad():
        [model(input_tensor) for _ in range(INFERENCE_TIMES)]
    torch.cuda.synchronize()
    end = time.time() * 1000 - a
    return end / INFERENCE_TIMES


architectures = {
    (512, 1024, 10, 256) : "mlp_small.onnx",
    (1024, 2048, 10, 512) : "mlp_medium.onnx",
    (2048, 4096, 10, 1024) : "mlp_large.onnx",
    (4096, 8192, 10, 2048) : "mlp_huge.onnx",
}

for key, val in architectures.items():
    saveRandomModel(val, key[0], key[1], key[2])
    model = loadONNXW(val, key[0], key[1], key[2]).to("cuda")
    model.eval()
    average_inf_time = inference(model, key[0], key[3])

    print(f"Average pytorch inference time for model with architecure (input, hidden, output, batch) : {key} = {average_inf_time} ms")