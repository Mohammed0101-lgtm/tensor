#pragma once

#include <stdexcept>
#include <string>

class __type_error__ : public std::exception {
 private:
  std::string __imp_;

 public:
  explicit __type_error__(const std::string& __msg) : __imp_(__msg) {}
  explicit __type_error__(const char* __msg) : __imp_(__msg) {}

  __type_error__(const __type_error__& __other) noexcept : __imp_(__other.imp()) {}

  __type_error__& operator=(const __type_error__& __other) noexcept {
    if (this != &__other) __imp_ = __other.imp();

    return *this;
  }

  ~__type_error__() override = default;

  const char* what() const noexcept override { return "Type error occurred"; }

  std::string imp() const { return __imp_; }
};

class __index_error__ : public std::exception {
 private:
  std::string __imp_;

 public:
  explicit __index_error__(const std::string& __msg) : __imp_(__msg) {}
  explicit __index_error__(const char* __msg) : __imp_(__msg) {}

  __index_error__(const __index_error__& __other) noexcept : __imp_(__other.imp()) {}

  __index_error__& operator=(const __index_error__& __other) noexcept {
    if (this != &__other) __imp_ = __other.imp();

    return *this;
  }

  ~__index_error__() override = default;

  const char* what() const noexcept override { return "Index error occured"; }

  std::string imp() const { return __imp_; }
};

class __shape_error__ : public std::exception {
 private:
  std::string __imp_;

 public:
  explicit __shape_error__(const std::string& __msg) : __imp_(__msg) {}
  explicit __shape_error__(const char* __msg) : __imp_(__msg) {}

  __shape_error__(const __shape_error__& __other) noexcept : __imp_(__other.imp()) {}

  __shape_error__& operator=(const __shape_error__& __other) noexcept {
    if (this != &__other) __imp_ = __other.imp();

    return *this;
  }

  ~__shape_error__() override = default;

  const char* what() const noexcept override { return "Shape error occured"; }

  std::string imp() const { return __imp_; }
};

class __access_error__ : public std::exception {
 private:
  std::string __imp_;

 public:
  explicit __access_error__(const std::string& __msg) : __imp_(__msg) {}
  explicit __access_error__(const char* __msg) : __imp_(__msg) {}

  __access_error__(const __access_error__& __other) noexcept : __imp_(__other.imp()) {}

  __access_error__& operator=(const __access_error__& __other) noexcept {
    if (this != &__other) __imp_ = __other.imp();

    return *this;
  }

  ~__access_error__() override = default;

  const char* what() const noexcept override { return "Access error ocurred"; }

  std::string imp() const { return __imp_; }
};