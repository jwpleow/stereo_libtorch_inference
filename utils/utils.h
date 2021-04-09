#pragma once

#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <torch/script.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

namespace Utils{
    
    // combines left and right into one matrix
    void combineImages(const cv::Mat &left, const cv::Mat &right, cv::Mat &combined);

    /*
    Transforms given OpenCV mat image (should be CV_8UC3)
    to a normalised and padded torch::Tensor of size [1, C, H, W]
    and sends to device
    */
    void transformImage(const cv::Mat& input, torch::Tensor& output, const cv::Size& required_padding, torch::Device device);


    /*
    Exports given tensor. Can be loaded in python using 
    
    data_fp = "path/to/tensor.pt"
    model = torch.jit.load(data_fp)  # libtorch saved our tensor in a model, in version 1.6/1.7.
    x = list(model.parameters())[0]
    */
    void saveTorchTensor(const torch::Tensor& tens, const std::string& filename = "tensor.pt");

    // Prints the attributes of the tensor (e.g. size, datatype, device)
    void printTensorProperties(const torch::Tensor& tens);
    // Prints the attributes of the cv::Mat
    void printCvMatProperties(const cv::Mat& mat);
    
    /* 
    Converts cv::Mat to torch::Tensor (For Images)
        cv::Mat - MUST BE CONTIGUOUS DATA! Safest to .clone() the cv::Mat before converting it here 
                  Should be in CV_8UCx, with the default arrangement of dimensions [H, W, C]. 
        torch::Tensor - size [1, C, H, W], float, normalized to [0, 1]
    */
    void convertCvMatToTensor(const cv::Mat& mat, torch::Tensor& tens);

    /*
    simple conversion (without channels)
        torch::Tensor - should be float, [H, W]
        cv::Mat - CV_32F, [H, W]
    */
    void convertTensorToCvMat(const torch::Tensor& tens, cv::Mat& mat);

    /*
        Input tensor should be [1, C, H, W] and normalized to [0,1]
        Applies ImageNet normalization to the tensor:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    */
    void normalizeTensor(torch::Tensor &tens);

    /*
    Calculates the padding required to bring original_size to expected_size. 
    Note values in padding_needed will default to 0 if original size exceeds expected size
    */
    void calculatePadding(const cv::Size& original_size, const cv::Size& expected_size, cv::Size& padding_needed);

    /*
    Input tensor should be [1, C, H, W]
    Pads the tensor to have dimensions that are a multiple of <factor>, with padding added at the top and right side
    */
    void padTensor(const torch::Tensor& original, torch::Tensor& padded, int factor = 96);

    /*
    Input tensor should be [1, C, H, W]
    Pads the tensor by padding.width and padding.height, with padding added at the top and right side
    */
    void padTensor(const torch::Tensor &original, torch::Tensor &padded, const cv::Size &padding);

    /*
    unpad CvMat to original_size, assuming padding_added was added on the top right. Makes a deep copy
    */
    void unpadCvMat(const cv::Mat &padded, cv::Mat &unpadded, const cv::Size& original_size, const cv::Size& padding_added);

    // Prints datatype of the cv::Mat e.g. CV_32FC3
    // pass in the type like this: std::string type = getCvMatType(mat.type())
    std::string getCvMatType(int type);

    std::string GetDateTime();

    /*
    Simple FPSCounter that uses a moving average
    To use, initialise somewhere, and call tick() every frame
    */
    class FPSCounter
    {
    public:
        FPSCounter();
        virtual ~FPSCounter();

        // call init when ready to start measuring 
        void init();
        // prints avg_fps_
        void printAvgFps();
        // returns avg_fps_
        float getAvgFps();
        // tick. set print to true to print a message when fps updates
        void tick(bool print = false);


    protected:
        std::chrono::system_clock::time_point start_time_; // start time to count till 1 second

        float avg_fps_ = 10; // average fps so far, initialise with an estimate
        int frames1sec_ = 0; // frames so far in the last second
        int frames_all_ = 0; // total frames seen so far
    };

    
    /*
    Timer from https://gist.github.com/gongzhitaao/7062087
    Usage:
    Utils::Timer tmr;
    // code
    double t = tmr.elapsed();
    std::cout << t << std::endl;

    tmr.reset();
    // code
    t = tmr.elapsed();
    std::cout << t << std::endl;
    */
    class Timer
    {
    public:
        Timer() : beg_(clock_::now()) {}
        void reset() { beg_ = clock_::now(); }
        double elapsed() const
        {
            return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
        }

    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double, std::ratio<1>> second_;
        std::chrono::time_point<clock_> beg_;
    };
} // namespace Utils