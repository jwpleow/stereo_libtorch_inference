cmake_minimum_required(VERSION 3.0)
project(torch_inference)

## cmake commands
## if using -D_GLIBCXX_USE_CXX11_ABI=1 libtorch... (Recommended, unless you are also using precxx11 abi opencv, etc) use:
# cmake -DCMAKE_PREFIX_PATH=/home/joel/Downloads/libtorch1.7 ..
## pytorch from conda has -D_GLIBCXX_USE_CXX11_ABI=0
# cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..



# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(CMAKE_FIND_DEBUG_MODE 1)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/dcn_cpp_plugin)
add_executable(${PROJECT_NAME} main.cpp InferClient.cpp Camera.cpp utils/utils.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed dcn_v2_cuda_forward_v2)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch include path: ${TORCH_LIBRARIES}")

# ROS
set(ROS_FOUND FALSE)
if(DEFINED ENV{ROS_DISTRO})
  set(FOUND_ROS_DISTRO $ENV{ROS_DISTRO})
  set(ROS_FOUND TRUE)
else()
  message("ROS distro variable not set. Trying to figure it out...")
  set(AVAILABLE_ROS_VERSIONS "noetic;melodic;lunar;kinetic;jade;indigo")
  set(ROS_FOUND FALSE)
  foreach(version ${AVAILABLE_ROS_VERSIONS})
    if(NOT ROS_FOUND)
      find_path(ROS_H ros.h PATHS /opt/ros/${version}/include/ros)
      if(ROS_H)
        message("Found ros version ${version}")
        set(FOUND_ROS_DISTRO ${version})
        set(ROS_FOUND TRUE)
      endif()
    endif()
  endforeach()
endif()

if(ROS_FOUND)
    include_directories(/opt/ros/$ENV{ROS_DISTRO}/include /opt/ros/$ENV{ROS_DISTRO}/lib)

    add_executable(camera_node main_ros.cpp ROS_Input.cpp InferClient.cpp Camera.cpp utils/utils.cpp)
    target_link_libraries(camera_node -I/opt/ros/$ENV{ROS_DISTRO}/include -L/opt/ros/$ENV{ROS_DISTRO}/lib
    -lroscpp -lrostime -lrosconsole -lroscpp_serialization -lcv_bridge -limage_transport)
    target_link_libraries(camera_node "${TORCH_LIBRARIES}")
    target_link_libraries(camera_node -Wl,--no-as-needed dcn_v2_cuda_forward_v2)
    target_link_libraries(camera_node ${OpenCV_LIBS})
endif()