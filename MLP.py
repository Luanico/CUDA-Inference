import torch
import numpy as np
import onnx

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


INPUT_SIZE = 10
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1

def loadONNXW(filename):
    model = onnx.load(filename)


def main():


