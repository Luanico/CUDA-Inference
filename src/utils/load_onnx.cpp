#include "load_onnx.h"

#include <fstream>
#include <cassert>
#include <iostream> 
#include <vector>   
#include <string>    
#include <cstring>


std::vector<std::vector<float>> getWeights(const google::protobuf::RepeatedPtrField<onnx::TensorProto>& tensors)
{
    std::vector<std::vector<float>> weights(tensors.size());


    for (size_t i = 0; i < tensors.size(); i++)
    {
        weights.at(i).resize(tensors.at(i).raw_data().size() / sizeof(float));
        memcpy(weights.at(i).data(), tensors.at(i).raw_data().data(), tensors.at(i).raw_data().size());
    }

    return weights;
}

std::vector<std::vector<float>> getWeightsFromFile(char* filename)
{
    std::ifstream input(filename, std::ios::in | std::ios::binary);
    onnx::ModelProto model;
    model.ParseFromIstream(&input);
    ::onnx::GraphProto graph = model.graph();

    std::vector<std::vector<float>> weights = getWeights(graph.initializer());

    return weights;
}

