#include <vector>
#include "proto/onnx.pb.h"

std::vector<std::vector<float>> getWeights(const google::protobuf::RepeatedPtrField<onnx::TensorProto>& tensors);
std::vector<std::vector<float>> getWeightsFromFile(char* filename);