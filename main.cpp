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

    Utils::FPSCounter fps;
    fps.init();
    while (true)
    {
        cv::Mat left, right;
        cam.read(left, right);

        // left = cv::imread("../left.png");
        // right = cv::imread("../right.png");

        cv::imshow("Left", left);
        cv::imshow("Right", right);
        cv::Mat disparity;
        aanetInferClient.runInference(left, right, disparity);

        cv::Mat visDisparity;
        visDisparity = disparity / 200.;
        cv::imshow("Disparity", visDisparity);
        fps.tick();
        fps.printAvgFps();
        if (cv::waitKey(1) >= 0)
        {
            break;
        }
        
    }
  


    return 0;
}

