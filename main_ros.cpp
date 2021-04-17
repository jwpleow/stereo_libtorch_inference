#include <chrono>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"

#include "InferClient.h"
#include "Camera.h"
#include "utils/utils.h"
#include "ROS_Input.h"


void publishImage(const ros::Publisher& pub, const cv::Mat& image, const std::string& encoding, const ros::Time& timestamp)
{
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image img_msg;
    std_msgs::Header header;
    header.stamp = timestamp;
    img_bridge = cv_bridge::CvImage(header, encoding, image);
    img_bridge.toImageMsg(img_msg);

    pub.publish(img_msg);
}

// image should be 8FC3 [0, 255], and depthmap should be 32FC3 (xyz), and same size as image
void publishPointCloud(const ros::Publisher& pub, const cv::Mat& image, const cv::Mat& depthmap, const ros::Time& timestamp)
{
    // https://github.com/ros-perception/image_pipeline/blob/noetic/stereo_image_proc/src/nodelets/point_cloud2.cpp#L159
    // http://docs.ros.org/en/melodic/api/cartographer_ros/html/msg__conversion_8cc_source.html
    // xyz field and intensity [0, 1] field?
    sensor_msgs::PointCloud2 cloudmsg;
    cloudmsg.header.stamp = timestamp;
    cloudmsg.header.frame_id = "left_camera_link";
    cloudmsg.width = depthmap.cols;
    cloudmsg.height = depthmap.rows;
    cloudmsg.is_dense = true; // unless theres invalid points
    cloudmsg.is_bigendian = false;

    cloudmsg.row_step = depthmap.step;

    sensor_msgs::PointCloud2Modifier pcd_modifier(cloudmsg);
    // pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    pcd_modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32,
                                      "y", 1, sensor_msgs::PointField::FLOAT32,
                                      "z", 1, sensor_msgs::PointField::FLOAT32,
                                      "intensity", 1, sensor_msgs::PointField::FLOAT32);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloudmsg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloudmsg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloudmsg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloudmsg, "intensity");
    cloudmsg.data.resize(cloudmsg.height * cloudmsg.row_step);
    
    if (depthmap.rows != image.rows || depthmap.cols != image.cols ) std::cerr << "depthmap and image have diff dimensions!\n";

    cv::Mat_<cv::Vec3f> mat = depthmap;

    for (int r = 0; r < mat.rows; r++)
    {
        for (int c = 0; c < mat.cols; c++, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity)
        {
            *iter_x = mat(r, c)[0];
            *iter_y = mat(r, c)[1];
            *iter_z = mat(r, c)[2];
            *iter_intensity = image.at<float>(r, c);
        }
    }

    pub.publish(cloudmsg);
}


int main(int argc, char** argv)
{
    std::string calib_params_file = "../CalibParams_Stereo.yml";
    std::string model_path = "../aanet_gpu_model.pt";
    cv::Size expected_size(672, 384); // WxH
    torch::Device device(torch::kCUDA);

    std::string raw_camera_topic = "/camera";

    Camera::StereoRectifier rectifier(calib_params_file);

    ros::init(argc, argv, "camera_processor");

    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(4);
    spinner.start();

    ROS_Input ros_input(nh, raw_camera_topic);
    cv::Mat frame;
    int64_t timestamp;

    if (ros::ok())
        ROS_INFO("camera_processor node ok");

    while (ros::ok())
    {
        timestamp = ros_input.read(frame);

        cv::imshow("frame", frame);
        cv::waitKey(1);

    }
    ros::waitForShutdown();
    
    

    // // ros::Publisher left_image_pub = n.advertise<sensor_msgs::Image>("/camera/left/rectified", 2);
    // // ros::Publisher right_image_pub = n.advertise<sensor_msgs::Image>("/camera/right/rectified", 2);
    // // ros::Publisher left_disp_pub = n.advertise<sensor_msgs::Image>("/camera/left/depth", 2);
    // ros::Publisher point_cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/camera/left/pointcloud", 2);

    // ros::Rate rate(10.0);

    // uint32_t seq = 0;

    // cv::Mat left, right, left8U, right8U;
    // cv::Mat depthmap;

    // if (ros::ok())
    //     ROS_INFO("Camera publisher started on /camera/left/rectified, /camera/right/rectified, /camera/left/disparity");

    // while (ros::ok())
    // {
    //     client.getDepth(left, right, depthmap);

    //     // cv::extractChannel(left, leftMono_f, 0);
    //     // cv::extractChannel(right, rightMono, 0);
    //     // leftMono_f.convertTo(leftMono, CV_8U);
    //     // rightMono.convertTo(rightMono, CV_8U);

    //     auto timestamp = ros::Time::now() - ros::Duration(0.1); // TODO: this offsets the timestamp since I don't have a way to get the frame timestamp atm..
    //     // publishImage(left_image_pub, leftMono, sensor_msgs::image_encodings::MONO8, timestamp);
    //     // publishImage(right_image_pub, rightMono, sensor_msgs::image_encodings::MONO8, timestamp);
    //     // publishImage(left_disp_pub, depthmap, sensor_msgs::image_encodings::TYPE_32FC1, timestamp);
    //     // publishPointCloud(point_cloud_pub, leftMono_f, depthmap, timestamp);
        

    //     ros::spinOnce();
    //     rate.sleep();
    // }

    return 0;
}
