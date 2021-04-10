#include "InferClient.h"
#include "Camera.h"
#include "utils/utils.h"


int main()
{
    // test
    std::string capture_string = "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink";
    std::string calib_params_file = "../CalibParams_Stereo.yml";
    std::string model_path = "../aanet_gpu_model.pt";
    cv::Size expected_size(672, 384); // WxH
    torch::Device device(torch::kCUDA);

    Inference::CameraInferClient client(capture_string, calib_params_file, model_path, expected_size, device);

    cv::Mat left, right;
    cv::Mat disparity;

    Utils::FPSCounter fps;
    while (true)
    {
        client.getFeed(left, right, disparity);
        cv::imshow("Left", left);
        cv::imshow("Right", right);

        cv::Mat visDisparity;
        double min_disp_val, max_disp_val;
        cv::minMaxLoc(disparity, &min_disp_val, &max_disp_val);
        visDisparity = disparity / max_disp_val;
        cv::imshow("Disparity", visDisparity);

        fps.tick(true);

        if (cv::waitKey(1) >= 0)
        {
            break;
        }
        
    }
  


    return 0;
}

