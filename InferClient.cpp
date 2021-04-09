#include <chrono>

#include "InferClient.h"

#include "utils/utils.h"

namespace Inference
{

InferClient::InferClient(const std::string &model_path, const cv::Size& expected_size, torch::Device device)
: expected_size(expected_size), device(device)
{
    // load model
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
    if (left.size != right.size)
    {
        std::cerr << "Warning: left and right image input do not have equal size!\n";
        std::cerr << "Left image size: " << left.size;
        std::cerr << "Right image size: " << right.size;
    }
    if (left.rows > expected_size.height || left.cols > expected_size.width)
    {
        std::cerr << "Warning: image input has a size greater than the expected image size:";
        std::cerr << "Expected height: " << expected_size.height << ", Left image height: " << left.rows << "\n";
        std::cerr << "Expected width: " << expected_size.width << ", Left image width: " << left.cols << "\n";
    }

    // preprocessing
    torch::Tensor leftT, rightT;
    original_size = cv::Size(left.cols, left.rows);
    Utils::calculatePadding(original_size, expected_size, required_padding);
    Utils::transformImage(left, leftT, required_padding, device);
    Utils::transformImage(right, rightT, required_padding, device);

    std::vector<torch::jit::IValue> input;
    input.push_back(leftT);
    input.push_back(rightT);

    // predict
    auto output = model.forward(input).toTensor().cpu().squeeze();

    cv::Mat disparity_temp;
    Utils::convertTensorToCvMat(output, disparity_temp);
    Utils::unpadCvMat(disparity_temp, disparity, original_size, required_padding);
}

CameraInferClient::CameraInferClient(const std::string &capture_string, const std::string &stereo_calib_params_file, const std::string &model_path, const cv::Size &expected_size, torch::Device device)
    : cam(capture_string, stereo_calib_params_file), InferClient(model_path, expected_size, device)
{
}

CameraInferClient::~CameraInferClient()
{
}

void CameraInferClient::getDisparity(cv::Mat& disparity)
{
    cv::Mat left, right;
    cam.read(left, right);

    runInference(left, right, disparity);
}

} // namespace Inference