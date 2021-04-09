#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core.hpp>

#include "Camera.h"

namespace Inference
{
    class InferClient
    {
        public:

        InferClient(const std::string& model_path, const cv::Size& expected_size, torch::Device device);
        virtual ~InferClient();

        void runInference(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);

        private:
        // original image size in input
        cv::Size original_size;
        // expected image size to feed into the model
        cv::Size expected_size;
        // variable to store the padding required on the input
        cv::Size required_padding;

        torch::Device device;
        torch::jit::script::Module model;
    };

    class CameraInferClient : public InferClient
    {
        public:
        CameraInferClient(const std::string& capture_string, const std::string& stereo_calib_params_file, const std::string& model_path, const cv::Size& expected_size, torch::Device device);
        ~CameraInferClient();

        void getDisparity(cv::Mat& disparity);

        public:
        Camera::StereoCamera cam;

    };

} // namespace Inference