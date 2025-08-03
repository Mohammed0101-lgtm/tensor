#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #define TENSOR_LIBRARY_API __declspec(dllexport)
#else
  #define TENSOR_LIBRARY_API __attribute__((visibility("default")))
#endif

#define TENSOR_INLINE inline
#define TENSOR_NODISCARD [[nodiscard]]
#define TENSOR_NOEXCEPT noexcept