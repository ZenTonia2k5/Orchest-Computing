cmake_minimum_required(VERSION 3.14)
project(orchest_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(orchest_core SHARED dp_solver.cpp)

set_target_properties(orchest_core PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "orchest_core"
)
