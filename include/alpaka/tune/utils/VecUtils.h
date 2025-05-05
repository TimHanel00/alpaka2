//
// Created by tim on 05.05.25.
//

#ifndef VECUTILS_H
#define VECUTILS_H

#include <alpaka/Vec.hpp>

#include <algorithm> // for std::min / std::max
#include <cmath> // for std::abs
#include <cstdint> // for uint32_t

namespace alpaka::tune::utils
{
    template<typename T_Vec>
    auto min_element(T_Vec const& vec)
    {
        using T = typename T_Vec::type;
        T min = vec[0];
        for(uint32_t i = 1; i < T_Vec::dim(); ++i)
            min = std::min(min, vec[i]);
        return min;
    }

    template<typename T_Vec>
    auto max_element(T_Vec const& vec)
    {
        using T = typename T_Vec::type;
        T max = vec[0];
        for(uint32_t i = 1; i < T_Vec::dim(); ++i)
            max = std::max(max, vec[i]);
        return max;
    }
    template<typename T>
    struct unwrap;

    template<typename T, auto T_Dim, typename T_Storage>
    struct unwrap<alpaka::Vec<T, T_Dim, T_Storage>>
    {
        static constexpr auto Dim = T_Dim;
    };

    /*
     * avoids default construction of a tuple to retrieve the type (usually done with alpaka::getDim(T{}))
     */
    template<typename T_Vec>
    struct getDimFromTemplate
    {
        static constexpr auto Dim = unwrap<T_Vec>::Dim;
    };

    template<typename T_Vec>
    struct toRTime
    {
        using get = alpaka::Vec<
            typename T_Vec::type,
            getDimFromTemplate<T_Vec>::Dim,
            alpaka::ArrayStorage<typename T_Vec::type, getDimFromTemplate<T_Vec>::Dim>>;
    };

    template<typename T_Vec>
    auto anyTrue(T_Vec const& vec)
    {
        for(std::size_t i = 0; i < getDimFromTemplate<T_Vec>::Dim; ++i)
            if(vec[i])
                return true;
        return false;
    }

    template<typename T_Vec>
    auto anyFalse(T_Vec const& vec)
    {
        for(std::size_t i = 0; i < getDimFromTemplate<T_Vec>::Dim; ++i)
            if(vec[i])
                return false;
        return true;
    }

    template<typename T_Vec>
    auto allTrue(T_Vec const& vec)
    {
        return !anyFalse(vec);
    }

    template<typename T_Vec>
    auto allFalse(T_Vec const& vec)
    {
        return !anyTrue(vec);
    }

    template<typename T_Vec>
    auto abs(T_Vec const& vec)
    {
        using ValueType = typename T_Vec::type;
        using ResultVec = typename toRTime<T_Vec>::get;


        if constexpr(std::is_signed_v<ValueType>)
        {
            ResultVec result;
            for(uint32_t i = 0; i < getDimFromTemplate<T_Vec>::Dim; ++i)
            {
                result[i] = std::abs(vec[i]);
            }
            return result;
        }
        else
            return vec;
    }
} // namespace alpaka::tune::utils
#endif // VECUTILS_H
