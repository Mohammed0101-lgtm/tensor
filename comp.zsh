$(brew --prefix llvm)/bin/clang++ -std=c++20 test.cpp -o t -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -v