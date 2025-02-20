cmake_minimum_required(VERSION 3.14)
project(TensorLibrary)

set(CMAKE_CXX_STANDARD 20)

# Find required packages
find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# Define the test executable
add_executable(tensor_test tensor_test.cpp)

# Link the GTest library first
target_link_libraries(tensor_test PRIVATE GTest::gtest_main)

# Check and link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor_test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable testing and add the test
enable_testing()
add_test(NAME AllTestsInMain COMMAND tensor_test)