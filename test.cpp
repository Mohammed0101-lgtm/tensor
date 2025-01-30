#include "./src/tensor.hpp"


int main(void) {
  tensor<int> __t({1, 3, 3, 3, 3}, 5);
  __t = __t.asinh_();
  __t.print();

  return 0;
}

