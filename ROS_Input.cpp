#include "ROS_Input.h"
#include "utils/utils.h"

ROS_Input::ROS_Input(const ros::NodeHandle& nh, const std::string& camera_topic)
: nh_(nh), it_(nh_), frame_buffer(3)
{
    camera_sub_ = it_.subscribe(camera_topic, 1, [this](const sensor_msgs::Image::ConstPtr &msg){imageCallback(msg);});
}

ROS_Input::~ROS_Input()
{
    grabOn.store(false);
    if (frame_read_thread.joinable())
        frame_read_thread.join();
}

// get the last image frame, blocks until a frame is ready
int64_t ROS_Input::read(cv::Mat &frame)
{
    std::pair<cv::Mat, int64_t> temp;

    int n;
    while (frame_buffer.empty())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        n++;
        if ((n % 10) == 0 && n > 1)
            std::cout << "Frame buffer empty, waiting for image...\n";
    }

    {
        std::lock_guard<std::mutex> scopeLock(frame_read_lock); // todo: use a proper circular buffer
        temp = (*frame_buffer.front());
        frame = temp.first.clone();
        frame_buffer.pop();
    }
    return temp.second;
}

void ROS_Input::imageCallback(const sensor_msgs::Image::ConstPtr &msg)
{
    auto temp_timestamp = static_cast<int64_t>(msg->header.stamp.sec) * 1000 + static_cast<int64_t>(msg->header.stamp.nsec / 1000000);
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
    if (!frame_buffer.try_push(std::make_pair(temp, temp_timestamp)))
    {
        frame_buffer.pop(); // TODO: hmm... this is a terrible solution
        frame_buffer.push(std::make_pair(temp, temp_timestamp));
    }


}

