#include "src/tensor.hpp"  
#include <iostream>

// Custom struct for testing custom data types
struct CustomData
{
  int    x;
  double y;

  // Overload operators for tensor operations
  CustomData operator+(const CustomData& other) const { return {x + other.x, y + other.y}; }

  CustomData operator*(const CustomData& other) const { return {x * other.x, y * other.y}; }

  friend std::ostream& operator<<(std::ostream& os, const CustomData& data) {
    os << "{" << data.x << ", " << data.y << "}";
    return os;
  }
};

int main() {
  std::cout << "Testing Tensor Library\n";

  // Test with integers
  tensor<int> A({2, 2}, {1, 2, 3, 4});
  tensor<int> B({2, 2}, {5, 6, 7, 8});

  /*
  std::cout << "Tensor A:\n" << "\n";
  A.print();
  std::cout << "Tensor B:\n" << "\n";
  B.print();

  auto C = A + B;
  std::cout << "A + B:\n" << "\n";
  C.print();

  auto D = A * B;
  std::cout << "A * B:\n" << D << "\n";

  auto D = A.matmul(B);
  std::cout << "A * B:\n" << "\n";
  D.print();

  // Test with floating point numbers
  tensor<double> E({2, 2}, {1.1, 2.2, 3.3, 4.4});
  tensor<double> F({2, 2}, {5.5, 6.6, 7.7, 8.8});

  std::cout << "Tensor E:\n" << "\n";
  E.print();
  std::cout << "Tensor F:\n" << "\n";
  F.print();

  auto G = E + F;
  std::cout << "E + F:\n" << "\n";
  G.print();

  auto H = E.matmul(F);
  std::cout << "E * F:\n" << "\n";
  H.print();
  */

  // Test with custom data structure
  tensor<CustomData> I({2, 2}, {{1, 1.1}, {2, 2.2}, {3, 3.3}, {4, 4.4}});
  tensor<CustomData> J({2, 2}, {{5, 5.5}, {6, 6.6}, {7, 7.7}, {8, 8.8}});

  std::cout << "Tensor I:\n" << "\n";
  I.print();
  std::cout << "Tensor J:\n" << "\n";
  J.print();

  auto K = I + J;
  std::cout << "I + J:\n" << "\n";
  K.print();

  auto L = I.matmul(J);
  std::cout << "I * J:\n" << "\n";
  L.print();

  auto m = L.transpose();
  std::cout << "T of m:\n" << "\n";
  m.print();

  return 0;
}