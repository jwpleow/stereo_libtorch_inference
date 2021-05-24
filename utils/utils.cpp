#include "utils.h"

namespace Utils{

    void combineImages(const cv::Mat &left, const cv::Mat &right, cv::Mat &combined)
    {
        cv::hconcat(left, right, combined);
    }

    void transformImage(const cv::Mat &input, torch::Tensor &output, const cv::Size& required_padding, torch::Device device)
    {
        torch::Tensor temp;
        convertCvMatToTensor(input, temp);
        // normalizeTensor(temp);

        if (required_padding.height > 0 || required_padding.width > 0)
        {
            padTensor(temp, output, required_padding);
        }
        else 
        {
            output = temp;
        }
        
        output = output.to(device);
    }

    void saveTorchTensor(const torch::Tensor &tens, const std::string &filename)
    {
        torch::save(tens, filename);
    }


    void printTensorProperties(const torch::Tensor &tens)
    {
        std::cout << "Tensor Properties:\n"
                  << "Dimensions: " << tens.dim() << "\n"
                  << "Datatype: " << tens.dtype() << "\n"
                  << "Device: " << tens.device() << "\n"
                  << "Size: " << tens.sizes() << "\n"
                  << "Max value: " << torch::max(tens) << "\n"
                  << "Min value: " << torch::min(tens) << "\n"
                  << "Number of Elements: " << tens.numel() << "\n";
    }

    void printCvMatProperties(const cv::Mat& mat)
    {
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);

        cv::Scalar means = cv::mean(mat);

        std::cout << "OpenCV Mat Properties:\n"
                  << "Dimensions: " << mat.dims << "\n"
                  << "Datatype: " << getCvMatType(mat.type()) << "\n"
                  << "Size: " << mat.size << "\n"
                  << "Number of Elements: " << mat.total() << "\n"
                  << "Max value: " << maxVal << "\n"
                  << "Min value: " << minVal << "\n"
                  << "Means of each channel: " << means << "\n";
    }

    void convertCvMatToTensor(const cv::Mat &mat, torch::Tensor& tens)
    {
        tens = torch::zeros({mat.rows, mat.cols, mat.channels()});
        tens = torch::from_blob(mat.data, {1, mat.rows, mat.cols, mat.channels()}, at::kByte).clone();
        tens = tens.to(at::kFloat);
        tens = tens.permute({0, 3, 1, 2}); // rearrange to BxCxHxW
        // tens = tens.div(255.0);
    }

    void convertTensorToCvMat(const torch::Tensor &tens, cv::Mat &mat)
    {
        if (tens.dim() == 2)
        {
            mat = cv::Mat::zeros(tens.size(0), tens.size(1), CV_32F);
            std::memcpy(mat.data, tens.data_ptr(), sizeof(float) * tens.numel());
        }
        else if (tens.dim() == 4 && tens.size(1) == 3)
        { // TODO: this part does the wrong thing!?
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

    void padTensor(const torch::Tensor &original, torch::Tensor &padded, const cv::Size& padding)
    {
        padded = torch::zeros({1, original.size(1), original.size(2) + padding.height, original.size(3) + padding.width});
        padded = torch::nn::functional::pad(original, torch::nn::functional::PadFuncOptions({0, padding.width, padding.height, 0}));
    }

    void calculatePadding(const cv::Size& original_size, const cv::Size& expected_size, cv::Size& padding_needed)
    {
        if (expected_size.width - original_size.width < 0 || expected_size.height - original_size.height < 0)
        {
            std::cout << "Warning: expected input size is smaller than original image!\n";
        }
        padding_needed.height = std::max(expected_size.height - original_size.height, 0);
        padding_needed.width = std::max(expected_size.width - original_size.width, 0);
    }

    void unpadCvMat(const cv::Mat &padded, cv::Mat &unpadded, const cv::Size& original_size, const cv::Size& padding_added)
    {
        cv::Rect original_rect(0, padding_added.height, original_size.width, original_size.height); // since padded on the top and right
        unpadded = padded(original_rect).clone();
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
        
        oss << "C";
        oss << channels;
    
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

    void FPSCounter::init(std::string name)
    {
        name_ = name;
        start_time_ = std::chrono::system_clock::now();
    }

    void FPSCounter::printAvgFps()
    {
        std::cout << std::setprecision(2) << std::fixed << name_ << ": Average fps - " << avg_fps_ << std::endl;
    }

    float FPSCounter::getAvgFps()
    {
        return avg_fps_;
    }

    void FPSCounter::tick(bool print)
    {
        // if past 1 second, reset the start time and calculate the new fps
        if (int millisecond_count = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time_).count() > 1000)
        {
            start_time_ = std::chrono::system_clock::now();
            avg_fps_ = 0.8f * avg_fps_ + 0.2f * static_cast<float>(frames1sec_) * (1000.f / static_cast<float>(1000 + millisecond_count));
            frames1sec_ = 0;
            if (print) printAvgFps();
        }
        frames1sec_++;
    }

} // namespace Utils