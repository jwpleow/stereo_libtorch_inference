#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core.hpp>

namespace Inference
{
    class InferClient
    {
        public:

        InferClient(const std::string& model_path, torch::Device device);
        virtual ~InferClient();

        void runInference(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);


        private:

        torch::Device device;
        torch::jit::script::Module model;
    };

} // namespace Inference