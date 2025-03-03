//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
namespace alpaka::tune
{

// Placeholder for IdxRange

    // Interface for tunable objects

    // Generic Tuneable class template
    template <typename T = std::size_t,
              typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct Tuneable {
        T value;
        std::string name;
        std::optional<IdxRange<T_Begin, T_End, T_Stride>> idxRange;

        explicit Tuneable(T value, const std::string &name,
                          std::optional<IdxRange<T_Begin, T_End, T_Stride>> idxRange = std::nullopt)
            : value(value), name(name), idxRange(idxRange) {}

        std::string getName() const  { return name; }
    };

    // Specialized tunables
    template <typename T = std::size_t,typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct GridSizeTune : Tuneable<T> {
        T gridSize;

        explicit GridSizeTune(T initial_value = 64,
                              std::optional<IdxRange<T_Begin, T_End, T_Stride>> idxRange = std::nullopt)
            : gridSize(initial_value),Tuneable<T>(initial_value, "gridSize", idxRange){}
    };

    template <typename T = std::size_t,typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct ThreadBlockSizeTune : Tuneable<T> {
        T blockThreadSize;

        explicit ThreadBlockSizeTune(T initial_value = 256,
                                     std::optional<IdxRange<T_Begin, T_End, T_Stride>> idxRange = std::nullopt)
            : blockThreadSize(initial_value),Tuneable<T>(initial_value, "blockThreadSize", idxRange){}
    };

    template <typename T = std::size_t>
    struct BlockNGridSizeTune {
        GridSizeTune<T> gridSize;
        ThreadBlockSizeTune<T> blockThreadSize;

        explicit BlockNGridSizeTune(T grid_value = 64, T block_value = 256,
                                    std::optional<IdxRange<T, T, T>> idxRangeBlocks = std::nullopt,std::optional<IdxRange<T, T, T>> idxRangeThreads = std::nullopt)
            : gridSize(grid_value, idxRangeBlocks), blockThreadSize(block_value, idxRangeThreads) {}
    };


}

#endif //TUNEABLE_H
