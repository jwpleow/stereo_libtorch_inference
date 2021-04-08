#include <chrono>

#include "imgInfer.h"

#include "utils/utils.h"

namespace Inference
{

InferClient::InferClient(const std::string &model_path, torch::Device device)
: device(device)
{
    try
    {
        model = torch::jit::load(model_path);
        std::cout << model_path << " loaded successfully.\n";
    }
    catch (const c10::Error &e)
    {
        std::ostringstream oss;
        oss << "error loading the model " << model_path << "\n";
        throw std::runtime_error(oss.str());
    }
    model.to(device);
}

InferClient::~InferClient()
{
}

void InferClient::runInference(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity)
{
    torch::Tensor leftT, rightT;
    Utils::transformImage(left, leftT, device);
    Utils::transformImage(right, rightT, device);

    std::vector<torch::jit::IValue> input;
    input.push_back(leftT);
    input.push_back(rightT);

    // predict
    auto output = model.forward(input).toTensor().cpu().squeeze();

    Utils::convertTensorToCvMat(output, disparity);
}



} // namespace Inference