cmake_minimum_required(VERSION 3.1)
project(Tekken7_RL_CPP)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Torch (LibTorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files for the C++ components
add_library(health_bar_cnn SHARED health_bar_cnn.cpp)
add_library(ppo_agent SHARED ppo_agent.cpp)
add_library(replay_buffer SHARED replay_buffer.cpp)

# Link libraries
target_link_libraries(health_bar_cnn ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_link_libraries(ppo_agent ${TORCH_LIBRARIES})
target_link_libraries(replay_buffer)

# Specify the output directory for shared libraries
set_target_properties(health_bar_cnn PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(ppo_agent PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(replay_buffer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

