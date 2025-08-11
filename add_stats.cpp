#include "src/tensor.hpp"
#include <chrono>
#include <fstream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>


void write_to_csv(const std::string line, const std::string& filename) {
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open())
  {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }

  file << line << std::endl;

  file.close();
}

int main() {
  std::string filename = "addition_test_results.csv";
  std::string header   = "size,my_library,eigen_library,x_tensor_library";

  unsigned long long height     = 100;
  unsigned long long width      = 100;
  unsigned long long added_size = 100;

  write_to_csv(header, filename);

  // testing loop
  for (int i = 0, num_iterations = 120; i < num_iterations; ++i)
  {
    unsigned long long size = height * width;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Iteration: " << i + 1 << "/" << num_iterations << std::endl;
    std::cout << "Testing with size: " << size << std::endl;

    // xtensor testing
    xt::xarray<float> xtensor1 = xt::random::rand<float>({height, width});
    xt::xarray<float> xtensor2 = xt::random::rand<float>({height, width});

    auto xt_warmup = xtensor1 + xtensor2;

    auto                       xt_start  = std::chrono::high_resolution_clock::now();
    volatile xt::xarray<float> xt_result = xtensor1 + xtensor2;
    auto                       xt_end    = std::chrono::high_resolution_clock::now();

    // eigen testing
    Eigen::array<Eigen::Index, 2> shape({static_cast<long>(height), static_cast<long>(width)});
    Eigen::Tensor<float, 2>       rand_tensor(shape);
    Eigen::Tensor<float, 2>       rand_tensor2(shape);

    rand_tensor.setRandom();
    rand_tensor2.setRandom();
    auto warmup = (rand_tensor + rand_tensor2).eval();

    auto                             start  = std::chrono::high_resolution_clock::now();
    volatile Eigen::Tensor<float, 2> result = (rand_tensor + rand_tensor2).eval();
    auto                             end    = std::chrono::high_resolution_clock::now();

    // my library testing
    tensor<float> my_tensor({height, width});
    tensor<float> my_tensor2({height, width});

    my_tensor.randomize_(my_tensor.shape());
    my_tensor2.randomize_(my_tensor2.shape());

    auto                   my_warmup = my_tensor + my_tensor2;
    auto                   my_start  = std::chrono::high_resolution_clock::now();
    volatile tensor<float> my_result = my_tensor + my_tensor2;
    auto                   my_end    = std::chrono::high_resolution_clock::now();

    auto eigen_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto my_duration    = std::chrono::duration_cast<std::chrono::microseconds>(my_end - my_start).count();
    auto xt_duration    = std::chrono::duration_cast<std::chrono::microseconds>(xt_end - xt_start).count();

    std::string ling = std::to_string(height * width) + "," + std::to_string(my_duration) + ","
                     + std::to_string(eigen_duration) + "," + std::to_string(xt_duration);

    write_to_csv(ling, filename);

    std::cout << "Eigen duration      = " << eigen_duration << " microseconds" << ", "
              << "My library duration = " << my_duration << " microseconds" << ", "
              << "xtensor duration    = " << xt_duration << " microseconds" << std::endl;

    std::cout << "----------------------------------------" << std::endl;

    width += added_size;
    height += added_size;
  }

  return 0;
}