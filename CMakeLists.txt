cmake_minimum_required(VERSION 3.14)
project(TensorLibrary)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Find required packages
set(GTest_DIR /opt/homebrew/Cellar/googletest/1.16.0)  # Adjust if needed
find_package(GTest REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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