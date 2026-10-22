cmake_minimum_required(VERSION 4.3)
project(Lab_2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(Lab_2 main.cu)

set_target_properties(Lab_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
