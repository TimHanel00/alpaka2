//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include <string>
namespace alpaka::tune
{

    //TODO add standard constructor for Tunable (no string name) infer name later based on Kernel signature and Arguement positioon
    template<typename T= std::size_t>
    struct Tuneable
    {
        T& value;
        std::string name;

        explicit Tuneable(T& value, const std::string& name) : value(value), name(name) {}
    };
    template<typename T= std::size_t>
    struct GridSizeTune : Tuneable<T>
    {
        T gridSize;

        explicit GridSizeTune(T initial_value = 64)
            : Tuneable<T>(gridSize, "gridSize"), gridSize(initial_value) {}
    };
    template<typename T = std::size_t>  // Default to std::size_t
    struct BlockThreadSizeTune : Tuneable<T>
    {
        T blockThreadSize;

        BlockThreadSizeTune(T initial_value = T(256))  // Default to 256
            : Tuneable<T>(blockThreadSize, "blockThreadSize"), blockThreadSize(initial_value) {}
    };

    template<typename T=std::size_t>
    struct BlockNGridSizeTune
    {
        GridSizeTune<T> gridSize;
        BlockThreadSizeTune<T> blockThreadSize;

        explicit BlockNGridSizeTune(T grid_value = 64, T block_value = 256)
            : gridSize(grid_value), blockThreadSize(block_value) {}
    };

}

#endif //TUNEABLE_H
