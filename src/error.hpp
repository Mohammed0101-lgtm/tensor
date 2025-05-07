#pragma once  // Ensures the file is only included once during compilation (include guard)

#include <stdexcept>  // For std::exception base class
#include <string>     // For using std::string

// Custom exception class for type-related errors (e.g., mismatched data types)
class type_error: public std::exception
{
   private:
    std::string imp_;  // Stores the detailed error message

   public:
    // Constructors for initializing the error message
    explicit type_error(const std::string& msg) :
        imp_(msg) {}
    explicit type_error(const char* msg) :
        imp_(msg) {}

    // Copy constructor for safely duplicating error objects
    type_error(const type_error& other) noexcept :
        imp_(other.imp()) {}

    // Copy assignment operator with self-assignment check
    type_error& operator=(const type_error& other) noexcept {
        if (this != &other)
            imp_ = other.imp();
        return *this;
    }

    // Override base class destructor
    ~type_error() override = default;

    // Override `what()` to return a generic error message (not the detailed message)
    const char* what() const noexcept override { return "Type error occurred"; }

    // Provides access to the stored detailed message
    const std::string& imp() const { return imp_; }
};

// Custom exception class for out-of-range or invalid index errors
class index_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit index_error(const std::string& msg) :
        imp_(msg) {}
    explicit index_error(const char* msg) :
        imp_(msg) {}

    index_error(const index_error& other) noexcept :
        imp_(other.imp()) {}

    index_error& operator=(const index_error& other) noexcept {
        if (this != &other)
            imp_ = other.imp();
        return *this;
    }

    ~index_error() override = default;

    const char* what() const noexcept override { return "Index error occured"; }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for shape mismatches in data structures (e.g., matrices)
class shape_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit shape_error(const std::string& msg) :
        imp_(msg) {}
    explicit shape_error(const char* msg) :
        imp_(msg) {}

    shape_error(const shape_error& other) noexcept :
        imp_(other.imp()) {}

    shape_error& operator=(const shape_error& other) noexcept {
        if (this != &other)
            imp_ = other.imp();
        return *this;
    }

    ~shape_error() override = default;

    const char* what() const noexcept override { return "Shape error occured"; }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for access violations (e.g., unauthorized access or invalid permissions)
class access_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit access_error(const std::string& msg) noexcept :
        imp_(msg) {}
    explicit access_error(const char* msg) noexcept :
        imp_(msg) {}

    access_error(const access_error& other) noexcept :
        imp_(other.imp()) {}

    access_error& operator=(const access_error& other) noexcept {
        if (this != &other)
            imp_ = other.imp();
        return *this;
    }

    ~access_error() override = default;

    const char* what() const noexcept override { return "Access error ocurred"; }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for operator misuse or failure (e.g., unsupported operations)
class operator_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit operator_error(const std::string& msg) noexcept :
        imp_(msg) {}
    explicit operator_error(const char* msg) noexcept :
        imp_(msg) {}

    operator_error(const operator_error& other) noexcept :
        imp_(other.imp()) {}

    operator_error& operator=(const operator_error& other) noexcept {
        if (this != &other)
            imp_ = other.imp();
        return *this;
    }

    ~operator_error() override = default;

    const char* what() const noexcept override { return "Operator error occured"; }

    const std::string& imp() const { return imp_; }
};