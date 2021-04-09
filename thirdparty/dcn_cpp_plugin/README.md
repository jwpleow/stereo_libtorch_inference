# DCN C PLUS CPLUS PLUGIN
Pytorch 1.7 only!
from https://github.com/xi11xi19/CenterNet2TorchScript, fixed
(For exporting use only as it only has .forward()?)
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
```
(note, if you build like the above, its built with precxx11 ABI)
## Usage
### in Python
```
import torch
torch.ops.load_library("build/libdcn_v2_cuda_forward_v2.so")
```

### in C++
Link it in your CMakeLists.txt (look in `aanet/deploy/CMakeLists.txt` for an example):
```
add_subdirectory(dcn_cpp_plugin) 
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed dcn_v2_cuda_forward_v2)
```