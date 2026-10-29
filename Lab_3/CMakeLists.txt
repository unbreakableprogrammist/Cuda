cmake_minimum_required(VERSION 4.3)
project(Lab_3 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Lab_3 main.cu)

set_target_properties(Lab_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
