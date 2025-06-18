#pragma once  // Ensures the file is only included once during compilation (include guard)

#include <stdexcept>  // For std::exception base class
#include <string>     // For using std::string


namespace error {

// Custom exception class for type-related errors (e.g., mismatched data types)
class type_error: public std::exception
{
   private:
    std::string imp_;  // Stores the detailed error message

   public:
    // Constructors for initializing the error message
    explicit type_error(const std::string& msg) :
        imp_("Type error occurred" + msg) {}
    explicit type_error(const char* msg) :
        imp_("Type error occurred" + std::string(msg)) {}

    // Copy constructor for safely duplicating error objects
    type_error(const type_error& other) noexcept :
        imp_(other.imp()) {}

    // Copy assignment operator with self-assignment check
    type_error& operator=(const type_error& other) noexcept {
        if (this != &other)
        {
            imp_ = other.imp();
        }

        return *this;
    }

    // Override base class destructor
    ~type_error() override = default;

    // Override `what()` to return a generic error message (not the detailed message)
    const char* what() const noexcept override { return imp_.c_str(); }

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
        imp_("Index error occured" + msg) {}

    explicit index_error(const char* msg) :
        imp_("Index error occured" + std::string(msg)) {}

    index_error(const index_error& other) noexcept :
        imp_(other.imp()) {}

    index_error& operator=(const index_error& other) noexcept {
        if (this != &other)
        {
            imp_ = other.imp();
        }

        return *this;
    }

    ~index_error() override = default;

    const char* what() const noexcept override { return imp_.c_str(); }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for shape mismatches in data structures (e.g., matrices)
class shape_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit shape_error(const std::string& msg) :
        imp_("Shape error occured : " + msg) {}

    explicit shape_error(const char* msg) :
        imp_("shape error occured : " + std::string(msg)) {}

    shape_error(const shape_error& other) noexcept :
        imp_(other.imp()) {}

    shape_error& operator=(const shape_error& other) noexcept {
        if (this != &other)
        {
            imp_ = other.imp();
        }

        return *this;
    }

    ~shape_error() override = default;

    const char* what() const noexcept override { return imp_.c_str(); }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for access violations (e.g., unauthorized access or invalid permissions)
class access_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit access_error(const std::string& msg) noexcept :
        imp_("Access error ocurred" + msg) {}
    explicit access_error(const char* msg) noexcept :
        imp_("Access error ocurred" + std::string(msg)) {}

    access_error(const access_error& other) noexcept :
        imp_(other.imp()) {}

    access_error& operator=(const access_error& other) noexcept {
        if (this != &other)
        {
            imp_ = other.imp();
        }

        return *this;
    }

    ~access_error() override = default;

    const char* what() const noexcept override { return imp_.c_str(); }

    const std::string& imp() const { return imp_; }
};

// Custom exception class for operator misuse or failure (e.g., unsupported operations)
class operator_error: public std::exception
{
   private:
    std::string imp_;

   public:
    explicit operator_error(const std::string& msg) noexcept :
        imp_("Operator error occured" + msg) {}
    explicit operator_error(const char* msg) noexcept :
        imp_("Operator error occured" + std::string(msg)) {}

    operator_error(const operator_error& other) noexcept :
        imp_(other.imp()) {}

    operator_error& operator=(const operator_error& other) noexcept {
        if (this != &other)
        {
            imp_ = other.imp();
        }

        return *this;
    }

    ~operator_error() override = default;

    const char* what() const noexcept override { return imp_.c_str(); }

    const std::string& imp() const { return imp_; }
};

}  // namespace error