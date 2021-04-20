# Hmm

## Dependencies
```
ROS
libtorch 1.7 # e.g. wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip
```

## Publishing the camera and disparity feeds to ROS
expects ImageTransport mono8 stereo frame on `/stereo/image`
```
source /opt/ros/noetic/setup.bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=path/to/libtorch1.7 ..
make
# roscore
./camera_publisher
```
publishes the pointcloud to `/stereo/pointcloud`