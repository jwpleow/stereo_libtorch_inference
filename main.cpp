#include "imgInfer.h"
#include "camera.h"
#include "utils/utils.h"


int main()
{
    // test
    std::string capture_string = "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink";
    torch::Device device(torch::kCUDA);

    Camera::StereoCamera cam(capture_string, "../CalibParams_Stereo.yml");
    Inference::InferClient aanetInferClient("../aanet_gpu_model.pt", device);

    while (true)
    {
        cv::Mat left, right;
        // cv::Mat left = cv::imread("../left.png");
        // cv::Mat right = cv::imread("../right.png");
        cam.read(left, right);
        cv::imshow("Left", left);
        cv::imshow("Right", right);
        cv::Mat disparity;
        aanetInferClient.runInference(left, right, disparity);

        cv::Mat visDisparity;
        double min_disp_val, max_disp_val;
        cv::minMaxLoc(disparity, &min_disp_val, &max_disp_val);
        visDisparity = disparity / max_disp_val;
        cv::imshow("Disparity", visDisparity);

        if (cv::waitKey(100) >= 0)
        {
            break;
        }
        
    }
  


    return 0;
}

