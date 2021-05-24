#include <chrono>

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "sensor_msgs/image_encodings.h"
#include "std_msgs/Header.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include <camera_info_manager/camera_info_manager.h>

#include "InferClient.h"
#include "Camera.h"
#include "utils/utils.h"
#include "ROS_Input.h"

void publishImage(const image_transport::Publisher& pub, const cv::Mat& image, const std::string& encoding, ros::Time& timestamp)
{
    sensor_msgs::ImagePtr msg;
    std_msgs::Header header;
    header.stamp = timestamp;
    header.frame_id = "left_camera_link";
    msg = cv_bridge::CvImage(header, encoding, image).toImageMsg();
    pub.publish(msg);
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
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    // pcd_modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32,
    //                                   "y", 1, sensor_msgs::PointField::FLOAT32,
    //                                   "z", 1, sensor_msgs::PointField::FLOAT32,
    //                                   "intensity", 1, sensor_msgs::PointField::FLOAT32);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloudmsg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloudmsg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloudmsg, "z");
    // sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloudmsg, "intensity");
    cloudmsg.data.resize(cloudmsg.height * cloudmsg.row_step);
    
    if (depthmap.rows != image.rows || depthmap.cols != image.cols ) std::cerr << "depthmap and image have diff dimensions!\n";

    cv::Mat_<cv::Vec3f> mat = depthmap;

    for (int r = 0; r < mat.rows; r++)
    {
        for (int c = 0; c < mat.cols; c++, ++iter_x, ++iter_y, ++iter_z) // ++iter_intensity
        {
            *iter_x = mat(r, c)[0];
            *iter_y = mat(r, c)[1];
            *iter_z = mat(r, c)[2];
            // *iter_intensity = image.at<float>(r, c);
        }
    }

    pub.publish(cloudmsg);
}


int main(int argc, char** argv)
{
    // inputs
    std::string calib_params_file = "../CalibParams_Stereo.yml";
    std::string model_path = "../bgnet_plus_model.pt";
    cv::Size expected_size(640, 384); // WxH
    torch::Device device(torch::kCUDA);

    std::string raw_camera_topic = "/stereo/image";
    std::string pointcloud_topic = "/stereo/pointcloud";

    Camera::StereoRectifier rectifier(calib_params_file);
    Inference::InferClient infer_client(model_path, expected_size, device);

    // init ros stuff
    ros::init(argc, argv, "camera_processor");
    
    ros::NodeHandle nh;
    ros::Rate rate(30.0);
    ros::AsyncSpinner spinner(4);
    spinner.start();

    image_transport::ImageTransport it(nh);

    ROS_Input ros_input(nh, raw_camera_topic);
    // ros::Publisher point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(pointcloud_topic, 2);
    // image_transport::CameraPublisher camera_pub = it.advertiseCamera(camera_pub_topic, 1);
    image_transport::Publisher left_rgb_rect_pub = it.advertise("/left/image_rect_color", 1);
    image_transport::Publisher right_rgb_rect_pub = it.advertise("/right/image_rect_color", 1);
    image_transport::Publisher depth_pub = it.advertise("/depth/image_rect", 1);
    // ros::Publisher depth_camera_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/depth/camera_info", 1);
    ros::Publisher left_camera_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/left/camera_info", 1);
    ros::Publisher right_camera_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/right/camera_info", 1);

    // camera info
    camera_info_manager::CameraInfoManager left_cam_info_mgr(nh, "arducamL", "file:///home/joel/Desktop/TorchInference/camera_info_left_ros.yaml");
    camera_info_manager::CameraInfoManager right_cam_info_mgr(nh, "arducamR", "file:///home/joel/Desktop/TorchInference/camera_info_right_ros.yaml");
    sensor_msgs::CameraInfo left_cam_info;
    sensor_msgs::CameraInfo right_cam_info;

    cv::Mat frame, frameC3, left_temp, right_temp, left, right, left_C3, right_C3;
    cv::Mat disparity, disparity_vis, pointcloud;
    ros::Time timestamp;

    if (ros::ok())
        ROS_INFO("camera_node ok. Pointcloud published on %s", pointcloud_topic.c_str());
    
    while (ros::ok())
    {
        timestamp = ros_input.read(frame); // CV_8UC1
        rectifier.splitImage(frame, left_temp, right_temp);
        rectifier.rectify(left_temp, right_temp, left, right);

        // manual crop, TODO: better more generalisable solution?
        cv::Rect newRect(0, 0, left.size[1], expected_size.height);
        left = left(newRect).clone();
        right = right(newRect).clone();
        // cv::fastNlMeansDenoising(left, left, 2.0f, 5, 9);
        // cv::fastNlMeansDenoising(right, right, 2.0f, 5, 9);

        cv::cvtColor(left, left_C3, cv::COLOR_GRAY2RGB);
        cv::cvtColor(right, right_C3, cv::COLOR_GRAY2RGB);



        infer_client.runInference(left, right, disparity);

        // // get pointcloud
        cv::reprojectImageTo3D(disparity, pointcloud, rectifier.stereo_calib_params.Q, true, -1); // -1 outputs CV_32F, will be reprojected to left camera's rectified coord system

        // publishPointCloud(point_cloud_pub, left, pointcloud, timestamp);

        // republish rectified camera feed as RGB with camerainfo, and depth image
        cv::Mat depth_image(pointcloud.rows, pointcloud.cols, CV_32F);
        cv::extractChannel(pointcloud, depth_image, 2);
        depth_image = depth_image.clone();

        publishImage(left_rgb_rect_pub, left_C3, "rgb8", timestamp);
        publishImage(right_rgb_rect_pub, right_C3, "rgb8", timestamp);
        publishImage(depth_pub, depth_image, "32FC1", timestamp);
        
        left_cam_info = left_cam_info_mgr.getCameraInfo();
        left_cam_info.header.stamp = timestamp;
        right_cam_info = right_cam_info_mgr.getCameraInfo();
        right_cam_info.header.stamp = timestamp;
        left_cam_info.header.frame_id = "left_camera_link";
        right_cam_info.header.frame_id = "right_camera_link";
        left_camera_info_pub.publish(left_cam_info);
        right_camera_info_pub.publish(right_cam_info);
        // depth_camera_info_pub.publish(left_cam_info);

        // visualisation
        double min_disp_val, max_disp_val;
        cv::minMaxLoc(disparity, &min_disp_val, &max_disp_val);
        disparity_vis = disparity / max_disp_val;
        cv::imshow("Left", left_C3);
        cv::imshow("Right", right_C3);
        cv::imshow("Disparity", disparity_vis);

        rate.sleep();
        if (cv::waitKey(1) >= 0) break;
    }
    spinner.stop();

    return 0;
}
