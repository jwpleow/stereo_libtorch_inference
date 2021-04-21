#pragma once

#include <string>
#include <iostream>
#include <mutex>
#include <thread>


#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include "thirdparty/SPSCQueue.h"

class ROS_Input
{
    public:
    ROS_Input(const ros::NodeHandle& nh, const std::string& camera_topic);
    virtual ~ROS_Input();

    // get the last left & right image
    ros::Time read(cv::Mat& frame);

    private:
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

    std::string camera_topic_;

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber camera_sub_;

    cv::Mat last_frame;
    ros::Time last_timestamp;
    rigtorp::SPSCQueue<std::pair<cv::Mat, ros::Time>> frame_buffer; // pair of image and timestamp in ms (epoch)

    std::thread frame_read_thread;
    std::mutex frame_read_lock;
    std::atomic<bool> grabOn;
};