rm -rf build
cmake -B build
cmake --build build
./build/tensor_test
