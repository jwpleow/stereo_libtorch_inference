# Hmm

remove the ROS bit in the cmake if you're not using the ros bit

## Publishing the camera and disparity feeds to ROS

```
source /opt/ros/noetic/setup.bash
mkdir build
cd build
cmake ..
make
# roscore
./camera_publisher
```