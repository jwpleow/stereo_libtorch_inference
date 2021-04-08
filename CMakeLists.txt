cmake_minimum_required(VERSION 3.0)
project(torch_inference)

## cmake commands
## pytorch from conda has -D_GLIBCXX_USE_CXX11_ABI=0
# cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
## if using -D_GLIBCXX_USE_CXX11_ABI=1 libtorch... use:
# cmake -DCMAKE_PREFIX_PATH=/home/asdf/Downloads/libtorch1.7 ..
## build (can use make too) 
# cmake --build . --config Release

# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
# set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# set(CMAKE_FIND_DEBUG_MODE 1)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/dcn_cpp_plugin)
add_executable(${PROJECT_NAME} main.cpp imgInfer.cpp camera.cpp utils/utils.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed dcn_v2_cuda_forward_v2)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "Torch include path: ${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
