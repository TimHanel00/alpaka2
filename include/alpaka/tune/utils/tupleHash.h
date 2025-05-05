//
// Created by tim on 05.05.25.
//

#ifndef TUPLEHASH_H
#define TUPLEHASH_H
#include <alpaka/alpaka.hpp>
#include <tuple>
#include <unordered_map>
#include <functional>
#include <iostream>

inline void hash_combine(std::size_t& seed) {}

template<typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, const Rest&... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

// Hash for alpaka::Vec
template<typename T, uint32_t Dim, typename Storage>
struct std::hash<alpaka::Vec<T, Dim, Storage>>
{
    std::size_t operator()(const alpaka::Vec<T, Dim, Storage>& v) const
    {
        std::size_t seed = 0;
        for(uint32_t i = 0; i < Dim; ++i)
            hash_combine(seed, v[i]);
        return seed;
    }
};

// Hash for std::tuple
template<typename Tuple, std::size_t Index = std::tuple_size<Tuple>::value - 1>
struct TupleHashHelper
{
    static void apply(std::size_t& seed, const Tuple& tuple)
    {
        TupleHashHelper<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template<typename Tuple>
struct TupleHashHelper<Tuple, 0>
{
    static void apply(std::size_t& seed, const Tuple& tuple)
    {
        hash_combine(seed, std::get<0>(tuple));
    }
};

struct TupleHash
{
    template<typename... Args>
    std::size_t operator()(const std::tuple<Args...>& tuple) const
    {
        std::size_t seed = 0;
        TupleHashHelper<std::tuple<Args...>>::apply(seed, tuple);
        return seed;
    }
};

#endif //TUPLEHASH_H
