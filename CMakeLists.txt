cmake_minimum_required(VERSION 3.14)
project(TensorLibrary)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Find required packages
set(GTest_DIR /opt/homebrew/Cellar/googletest/1.16.0) # Adjust if needed
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
add_executable(tensor_test
    tests/main.cpp
    tests/test_stride.cpp
    tests/test_internal.cpp
    tests/test_shape.cpp
    tests/test_constructors.cpp
    tests/tensor_test.cpp
    tests/test_tensorbool.cpp
    tests/test_getters.cpp
    tests/test_compare.cpp
    tests/test_types.cpp
    tests/test_operators.cpp
    tests/test_bit.cpp
    tests/test_simd.cpp
    tests/test_error.cpp
    tests/test_tensorbase.cpp
    tests/test_arithmetic.cpp
    tests/test_data.cpp
    tests/test_tensor.cpp
    tests/test_math/test_frac.cpp
    tests/test_math/test_dist.cpp
    tests/test_math/test_mod.cpp
    tests/test_math/test_max.cpp
    tests/test_math/test_pow.cpp
    tests/test_math/test_exp.cpp
    tests/test_math/test_sqrt.cpp
    tests/test_math/test_avg.cpp
    tests/test_math/test_abs.cpp
    tests/test_math/test_rem.cpp
    tests/test_math/test_lcm.cpp
    tests/test_math/test_trig/test_cos/test_cos.cpp
    tests/test_math/test_trig/test_cos/test_cosh.cpp
    tests/test_math/test_trig/test_cos/test_acosh.cpp
    tests/test_math/test_trig/test_cos/test_acos.cpp
    tests/test_math/test_trig/test_tan/test_atan.cpp
    tests/test_math/test_trig/test_tan/test_atanh.cpp
    tests/test_math/test_trig/test_tan/test_tan.cpp
    tests/test_math/test_trig/test_tan/test_tanh.cpp
    tests/test_math/test_trig/test_sin/test_asinh.cpp
    tests/test_math/test_trig/test_sin/test_sinc.cpp
    tests/test_math/test_trig/test_sin/test_sin.cpp
    tests/test_math/test_trig/test_sin/test_sinh.cpp
    tests/test_math/test_trig/test_sin/test_asin.cpp
    tests/test_math/test_log/test_log2.cpp
    tests/test_math/test_log/test_log.cpp
    tests/test_math/test_log/test_log10.cpp
    tests/test_linear/test_arg.cpp
    tests/test_linear/test_matmul.cpp
    tests/test_linear/test_sum.cpp
    tests/test_linear/test_clamp.cpp
    tests/test_linear/test_squeeze.cpp
    tests/test_linear/test_sigmoid.cpp
    tests/test_linear/test_transpose.cpp
    tests/test_linear/test_det.cpp
    tests/test_linear/test_product.cpp
    tests/test_linear/test_relu.cpp
    tests/test_linear/test_slice.cpp
)

# Link the GTest library first
target_link_libraries(tensor_test PRIVATE GTest::gtest_main)

# Check and link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor_test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Enable testing and add the test
enable_testing()
add_test(NAME AllTestsInMain COMMAND tensor_test)