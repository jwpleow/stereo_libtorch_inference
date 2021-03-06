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

        // CV_8UC3 images and CV_32FC1 disparity
        void getFeed(cv::Mat& left, cv::Mat& right, cv::Mat& disparity);
        // CV_8UC3 combined image instead with disparity
        void getFeed(cv::Mat& combined, cv::Mat& disparity);
        // CV_8UC3 images and CV_32FC1 depthmap (in metres) of the left image
        void getDepth(cv::Mat& left, cv::Mat& right, cv::Mat& depthmap);

        public:
        Camera::StereoCamera cam;

    };

} // namespace Inference