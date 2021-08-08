#include "ROS_Input.h"
#include "utils/utils.h"

ROS_Input::ROS_Input(const ros::NodeHandle& nh, const std::string& camera_topic)
: nh_(nh), it_(nh_)
{
    camera_sub_ = it_.subscribe(camera_topic, 1, [this](const sensor_msgs::Image::ConstPtr &msg){imageCallback(msg);});
    fps_counter.init("ROS camera feed");
}

ROS_Input::~ROS_Input()
{
    grabOn.store(false);
    if (frame_read_thread.joinable())
        frame_read_thread.join();
}

// get the last image frame, blocks until a frame is ready
ros::Time ROS_Input::read(cv::Mat &frame)
{
    ros::Time temp_time;

    int n;
    fps_counter.tick(true);
    {
        frame_read_lock.lock();
        while (last_frame.first.empty())
        {
            frame_read_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            n++;
            if ((n % 10) == 0 && n > 1)
                std::cout << "Frame buffer empty, waiting for image...\n";
            frame_read_lock.lock();
        }

        last_frame.first.copyTo(frame);
        temp_time = last_frame.second;
        frame_read_lock.unlock();
    }
 
    return temp_time;
}

void ROS_Input::imageCallback(const sensor_msgs::Image::ConstPtr &msg)
{
    ros::Time temp_timestamp = msg->header.stamp;
    cv::Mat temp;
    try
    {
        temp = (cv_bridge::toCvShare(msg, "mono8")->image).clone();
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Could not convert from '%s' to 'mono8'.", msg->encoding.c_str());
    }
    
    std::lock_guard<std::mutex> scope_lock(frame_read_lock);
    last_frame = std::make_pair(temp, temp_timestamp);
}

