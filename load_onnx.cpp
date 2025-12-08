#include <fstream>
#include <cassert>

#include "onnx.pb.h"

float* getWeights(const ::google::protobuf::RepeatedPtrField< ::onnx::TensorProto >& tensors)
{
    std::list<float> weights = {};
    for (auto t : tensors)
    {
        weights.push_back(t);
    }
}

int main(int argc, char* argv)
{
    std::ifstream input(argv[1], std::ios::in | std::ios::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&input);
    ::onnx::GraphProto graph = model.graph();

    std::list<float> weights = getWeights(graph.initializer());
    std::cout << weights;
    std::cout << std::endl;
}

