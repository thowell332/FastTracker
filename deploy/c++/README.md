#  FastTracker — C++ Version

A lightweight, CPU-only multi-object tracker inspired by **ByteTrack**, implemented entirely in C++.  


---

##  Dependencies

### Required Libraries

| Library | Purpose | Install Command (Ubuntu/Debian) |
|----------|----------|--------------------------------|
| **OpenCV ≥ 4.0** | Image loading, drawing, and I/O | ```bash sudo apt install libopencv-dev ``` |
| **Eigen ≥ 3.3** | Linear algebra for Kalman Filter | *(Header-only, bundled or install manually)* |
| **CMake ≥ 3.0** | Build system | ```bash sudo apt install cmake ``` |
| **g++ ≥ 8.0** | Compiler with C++11/17 support | ```bash sudo apt install build-essential ``` |

If you don’t have Eigen globally installed, just include it in your project directory and point CMake to it:

```cmake
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4-rc1/)
```