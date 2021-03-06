#include "Camera.h"


namespace Camera
{

StereoRectifier::StereoRectifier(const std::string& stereo_calib_file)
{
    loadStereoCalibParams(stereo_calib_file);
    initialiseStereoRectificationAndUndistortMap();
}

StereoRectifier::~StereoRectifier()
{
}

void StereoRectifier::rectify(const cv::Mat &input_left, const cv::Mat &input_right, cv::Mat &output_left, cv::Mat &output_right)
{
    // assert(!validMatchedRoi_1.empty() && !validMatchedRoi_2.empty());
    cv::remap(input_left, input_left, stereo_calib_params.undistort_map_1_1, stereo_calib_params.undistort_map_1_2, cv::INTER_LINEAR);
    output_left = input_left(stereo_calib_params.valid_matched_roi_1);
    cv::remap(input_right, input_right, stereo_calib_params.undistort_map_2_1, stereo_calib_params.undistort_map_2_2, cv::INTER_LINEAR);
    output_right = input_right(stereo_calib_params.valid_matched_roi_2);
}

void StereoRectifier::loadStereoCalibParams(const std::string &stereo_calib_file)
{
    // std::cout << "Loading stereo calibration parameter file " << stereo_calib_file << std::endl;
    cv::FileStorage fs(stereo_calib_file, cv::FileStorage::READ);

    if (fs.isOpened())
    {
        fs["image_width"] >> stereo_calib_params.image_width;
        fs["image_height"] >> stereo_calib_params.image_height;
        fs["left_camera_matrix"] >> stereo_calib_params.left_camera_matrix;
        fs["left_distortion_coefficients"] >> stereo_calib_params.left_dist_coeffs;
        fs["right_camera_matrix"] >> stereo_calib_params.right_camera_matrix;
        fs["right_distortion_coefficients"] >> stereo_calib_params.right_dist_coeffs;
        fs["R"] >> stereo_calib_params.R;
        fs["T"] >> stereo_calib_params.T;
        fs["E"] >> stereo_calib_params.E;
        fs["F"] >> stereo_calib_params.F;
        calibration_loaded = true;
        fs.release();
        std::cout << "Loaded stereo calibration file " << stereo_calib_file << std::endl;
    }
    else
    {
        std::stringstream oss;
        oss << "Error: Could not open calibration file " << stereo_calib_file << "\n";
        throw std::runtime_error(oss.str());
    }
}

void StereoRectifier::splitImage(const cv::Mat &original, cv::Mat &left, cv::Mat &right)
{
    static cv::Rect leftRect(0, 0, stereo_calib_params.image_width, stereo_calib_params.image_height);
    static cv::Rect rightRect(stereo_calib_params.image_width, 0, stereo_calib_params.image_width, stereo_calib_params.image_height);
    left = original(leftRect);
    right = original(rightRect);
}

void StereoRectifier::calculateMatchedRoi()
{
    // match y loc
    int new_y_loc = std::max(stereo_calib_params.valid_roi_1.y, stereo_calib_params.valid_roi_2.y);
    // x loc for right img
    int new_x_loc_right = std::max(stereo_calib_params.valid_roi_2.x, stereo_calib_params.image_width - stereo_calib_params.valid_roi_1.x - stereo_calib_params.valid_roi_1.width);
    int new_height = std::min(stereo_calib_params.valid_roi_1.height - (new_y_loc - stereo_calib_params.valid_roi_1.y), stereo_calib_params.valid_roi_2.height - (new_y_loc - stereo_calib_params.valid_roi_2.y));
    int new_width = std::min(std::min((stereo_calib_params.valid_roi_1.x + stereo_calib_params.valid_roi_1.width) - new_x_loc_right, stereo_calib_params.valid_roi_2.width), stereo_calib_params.image_width - stereo_calib_params.valid_roi_2.x - new_x_loc_right);

    stereo_calib_params.valid_matched_roi_1 = cv::Rect(stereo_calib_params.image_width - new_x_loc_right - new_width, new_y_loc, new_width, new_height);
    stereo_calib_params.valid_matched_roi_2 = cv::Rect(new_x_loc_right, new_y_loc, new_width, new_height);
}

void StereoRectifier::initialiseStereoRectificationAndUndistortMap(double alpha)
{
    assert((alpha >= 0.0 && alpha <= 1.0) || alpha == -1.0);
    if (calibration_loaded)
    {
        cv::Size image_size(stereo_calib_params.image_width, stereo_calib_params.image_height);
        cv::stereoRectify(stereo_calib_params.left_camera_matrix, stereo_calib_params.left_dist_coeffs, stereo_calib_params.right_camera_matrix, stereo_calib_params.right_dist_coeffs, image_size, stereo_calib_params.R, stereo_calib_params.T, stereo_calib_params.R1, stereo_calib_params.R2, stereo_calib_params.P1, stereo_calib_params.P2, stereo_calib_params.Q, cv::CALIB_ZERO_DISPARITY, alpha, image_size, &stereo_calib_params.valid_roi_1, &stereo_calib_params.valid_roi_2);

        // camera 1
        cv::initUndistortRectifyMap(stereo_calib_params.left_camera_matrix, stereo_calib_params.left_dist_coeffs, stereo_calib_params.R1, stereo_calib_params.P1, image_size, CV_16SC2, stereo_calib_params.undistort_map_1_1, stereo_calib_params.undistort_map_1_2); // could try speed comparison with CV_32FC1
        // camera 2
        cv::initUndistortRectifyMap(stereo_calib_params.right_camera_matrix, stereo_calib_params.right_dist_coeffs, stereo_calib_params.R2, stereo_calib_params.P2, image_size, CV_16SC2, stereo_calib_params.undistort_map_2_1, stereo_calib_params.undistort_map_2_2);
        calculateMatchedRoi();
    }
}

CameraBase::CameraBase(const std::string& capture_string)
{
    if (!openVideoCapture(capture_string))
    {
        throw std::runtime_error("Error: Could not open camera!\n");
    }

    grabOn.store(true);
    startUpdateThread();
}

CameraBase::~CameraBase()
{
    grabOn.store(false);
    if (frame_read_thread.joinable())
        frame_read_thread.join();
    video_capture.release();
}

bool CameraBase::openVideoCapture(const std::string &capture_string)
{
    std::cout << "Attempting to connect to VideoCapture...\n";
    if (video_capture.open(capture_string, cv::CAP_GSTREAMER))
    {
        // get one frame to find details
        video_capture >> last_frame;
        frame_width = last_frame.cols;
        frame_height = last_frame.rows;
        std::cout << "VideoCapture opened successfully. Mat type: " << Utils::getCvMatType(last_frame.type()) << ", Size: " << last_frame.size() << std::endl;

        return true;
    }
    else
    {
        return false;
    }
}

void CameraBase::update()
{
    fps_counter.init();
    // cv::Mat last_frame;
    cv::Mat temp_frame;
    while (grabOn.load() == true)
    {
        fps_counter.tick(true);

        video_capture >> temp_frame;
        std::lock_guard<std::mutex> scopeLock(frame_read_lock);
        last_frame = temp_frame;
        // long msec = video_capture.get(cv::CAP_PROP_POS_MSEC); // time of frame capture - perhaps may need to correct to unix https://answers.opencv.org/question/61099/is-it-possible-to-get-frame-timestamps-for-live-streaming-video-frames-on-linux/
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void CameraBase::startUpdateThread()
{
    frame_read_thread = std::thread(&CameraBase::update, this);
}

void CameraBase::read(cv::Mat& frame)
{
    std::lock_guard<std::mutex> scopeLock(frame_read_lock);
    frame = last_frame;
}


// StereoCamera
StereoCamera::StereoCamera(const std::string &capture_string, const std::string &stereo_calib_file)
    : CameraBase(capture_string), StereoRectifier(stereo_calib_file)
{
}

StereoCamera::~StereoCamera()
{
}

void StereoCamera::read(cv::Mat &left, cv::Mat &right)
{
    cv::Mat temp;
    cv::Mat left_temp, right_temp;
    
    int n;
    {
        frame_read_lock.lock();
        while (last_frame.empty())
        {
            frame_read_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            n++;
            if ((n % 10) == 0 && n > 1) std::cout << "Frame is empty, waiting for image...\n";
            frame_read_lock.lock();
        }

        last_frame.copyTo(temp);
        frame_read_lock.unlock();
    }
    
    splitImage(temp, left_temp, right_temp);
    // undistort and rectify, and crop
    rectify(left_temp, right_temp, left, right);
}

} // namespace Camera