#include "utils.h"

namespace Utils{

    void combineImages(const cv::Mat &left, const cv::Mat &right, cv::Mat &combined)
    {
        cv::hconcat(left, right, combined);
    }

    void transformImage(const cv::Mat &input, torch::Tensor &output, torch::Device device)
    {
        torch::Tensor temp;
        convertCvMatToTensor(input, temp);
        normalizeTensor(temp);
        // output = temp;
        padTensor(temp, output);

        output = output.to(device);
    }

    void saveTorchTensor(const torch::Tensor &tens, const std::string &filename)
    {
        torch::save(tens, filename);
    }


    void printTensorProperties(const torch::Tensor &tens)
    {
        std::cout << "Dimensions: " << tens.dim() << "\n"
                  << "Datatype: " << tens.dtype() << "\n"
                  << "Device: " << tens.device() << "\n"
                  << "Size: " << tens.sizes() << "\n"
                  << "Number of Elements: " << tens.numel() << "\n";
    }

    void printCvMatProperties(const cv::Mat& mat)
    {
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);
        std::cout << "Dimensions: " << mat.dims << "\n"
                  << "Datatype: " << getCvMatType(mat.type()) << "\n"
                  << "Size: " << mat.size << "\n"
                  << "Number of Elements: " << mat.numel() << "\n";
                  << "Max value in "
    }

    void convertCvMatToTensor(const cv::Mat &mat, torch::Tensor& tens)
    {
        tens = torch::zeros({mat.rows, mat.cols, mat.channels()});
        tens = torch::from_blob(mat.data, {1, mat.rows, mat.cols, mat.channels()}, at::kByte).clone();
        tens = tens.to(at::kFloat);
        tens = tens.permute({0, 3, 1, 2}); // rearrange to BxCxHxW
        tens = tens.div(255.0);
    }

    void convertTensorToCvMat(const torch::Tensor &tens, cv::Mat &mat)
    {
        if (tens.dim() == 2)
        {
            mat = cv::Mat::zeros(tens.size(0), tens.size(1), CV_32F);
            std::memcpy(mat.data, tens.data_ptr(), sizeof(float) * tens.numel());
        }
        else if (tens.dim() == 4 && tens.size(1) == 3)
        {
            mat = cv::Mat::zeros(tens.size(2), tens.size(3), CV_32FC3);
            torch::Tensor temp = tens.permute({0, 2, 3, 1}).clone();
            std::memcpy(mat.data, temp.data_ptr(), sizeof(float) * temp.numel());
        }
    }

    void normalizeTensor(torch::Tensor &tens)
    {
        tens[0][0] = tens[0][0].sub(0.485).div(0.229);
        tens[0][1] = tens[0][1].sub(0.456).div(0.224);
        tens[0][2] = tens[0][2].sub(0.406).div(0.225);
    }

    void padTensor(const torch::Tensor& original, torch::Tensor& padded, int factor)
    {
        static int ori_height = original.size(2);
        static int ori_width = original.size(3);
        static float ffactor = static_cast<float>(factor);
        static int img_height = static_cast<int>(std::ceil(static_cast<float>(ori_height) / ffactor) * ffactor);
        static int img_width = static_cast<int>(std::ceil(static_cast<float>(ori_width) / ffactor) * ffactor);

        padded = torch::zeros({1, original.size(1), img_height, img_width});
        if (ori_height < img_height || ori_width < img_width)
        {
            static int top_pad = img_height - ori_height;
            static int right_pad = img_width - ori_width;
            padded = torch::nn::functional::pad(original, torch::nn::functional::PadFuncOptions({0, right_pad, top_pad, 0}));
        }
        else
        {
            padded = original; // hmm...
        }
        
    }

    std::string getCvMatType(int type)
    {
        // taken from https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
        std::stringstream oss;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        int channels = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  oss << "CV_8U"; break;
            case CV_8S:  oss << "CV_8S"; break;
            case CV_16U: oss << "CV_16U"; break;
            case CV_16S: oss << "CV_16S"; break;
            case CV_32S: oss << "CV_32S"; break;
            case CV_32F: oss << "CV_32F"; break;
            case CV_64F: oss << "CV_64F"; break;
            default:     oss << "Undefined"; break;
        }
        
        if (channels > 1)
        {
            oss << "C";
            oss << channels;
        }
        
        return oss.str();
    }

    std::string GetDateTime()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t); // not thread safe!
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    // FPSCounter
    FPSCounter::FPSCounter()
    {
    }

    FPSCounter::~FPSCounter()
    {
    }

    void FPSCounter::init()
    {
        start_time_ = std::chrono::system_clock::now();
    }

    void FPSCounter::printAvgFps()
    {
        std::cout << std::setprecision(2) << std::fixed << "Average fps: " << avg_fps_ << std::endl;
    }

    float FPSCounter::getAvgFps()
    {
        return avg_fps_;
    }

    void FPSCounter::tick()
    {
        // if past 1 second, reset the start time and calculate the new fps
        if (int millisecond_count = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time_).count() > 1000)
        {
            start_time_ = std::chrono::system_clock::now();
            avg_fps_ = 0.8f * avg_fps_ + 0.2f * static_cast<float>(frames1sec_) * (1000.f / static_cast<float>(1000 + millisecond_count));
            frames1sec_ = 0;
        }
        frames1sec_++;
    }
} // namespace Utils