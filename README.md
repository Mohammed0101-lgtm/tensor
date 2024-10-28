# tensor

this is a C plus plus tensor library that enables the user (programmer) to create
tensors, transform them, apply operations on them, and so on..

## How does it work

Internally I thought about making it simple for the user to create a tensor like it was part 
of the CPP standard library, and providing many usefull methods to add tensors and multiply them,
transpose a tensor, retrieve a value from it and so fourth..
The tensor class is a template class that accepts any type of standard or user defined classes, though
you have to be carefull, some operations are only supported when the class complies with certain conditions.
When you create a tensor the memory is managed by the constructors and the destructors, so you don't have
to worry about it.

## Usage
```cpp
    #include "tensor.hpp"

    int main(void) {
        tensor<int> tens({3}, {1, 3, 4});
        tens.print();

        return 0;
    }
```
