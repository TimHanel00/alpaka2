//
// Created by tim on 28.04.25.
//

#ifndef PARTITIONING_H
#define PARTITIONING_H
#include "alpaka/Vec.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace alpaka::tune
{
    inline auto primeFactorize(std::size_t max)
    {
        std::size_t start = 2;
        std::vector<std::size_t> factors;
        while(start * start <= max)
        {
            if(max % start == 0)
            {
                factors.push_back(start);
                max /= start;
            }
            else
            {
                start++;
            }
        }
        if(max > 1)
        {
            factors.push_back(max);
        }
        std::sort(factors.rbegin(), factors.rend()); // sort descending
        return factors;
    }

    /*
     * this implements a partition method equally distributing prime factors across dims.
     * this guarentees that for vec.product() is exactly equal to max
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec primeFactorPartitioning(std::size_t max, T_vec const&)
    {
        using ValType = typename T_vec::type;
        auto vecFactors = primeFactorize(max);
        std::vector<ValType> distribute(T_vec::dim(), ValType(1));
        for(auto factor : vecFactors)
        {
            auto min_elem = std::min_element(distribute.begin(), distribute.end());
            *min_elem *= ValType(factor);
        }
        std::sort(distribute.begin(), distribute.end()); // sort ascending (since vec[0] is the slowest index)
        auto resultVec = Vec<ValType, T_vec::dim()>::all(1);
        for(std::size_t i = 0; i < T_vec::dim(); ++i)
        {
            resultVec[i] = distribute[i];
        }
        return resultVec;
    }

    /*
     * this implements a partition method where we take the ceiling of the nth root of the max for each dimension
     * this is a good strategy to distribute work equally does a lot of times more workers are used then necessary
     * very bad for distribution of the threadBlockSize, there primeFactorPartition should be used
     * T
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec ceilRootOverDimPartitioning(std::size_t max, T_vec const&)
    {
        using ValType = typename T_vec::type;
        // start with the 1s Vector
        auto resultVec = Vec<ValType, T_vec::dim()>::all(1);
        auto remainder = max;
        for(std::size_t i = 0; i < T_vec::dim(); ++i)
        {
            // using ceil-root heuristic to distribute mps across dimensions
            ValType split = std::max(ValType(1), static_cast<ValType>(std::pow(remainder, 1.0 / (T_vec::dim() - i))));
            resultVec = split;
            remainder /= split;
        }
        return resultVec;
    }

    /*
     * given a smaller ndim vector returns the the largest multiple of that ndim such that vec.product()<=max
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec multipleOfPartitioning(std::size_t max, T_vec vec)
    {
        using ValType = typename T_vec::type;
        // start with the 1s Vector
        auto resultVec = vec.toRT();
        auto initVec = resultVec;
        while(resultVec.product() <= max - initVec.product())
        {
            // round robin approach of incrementing dims since all of those combinations can be used in the backend

            resultVec += initVec;
        }
        return resultVec;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t ceilRootOverDimPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t primeFactorPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t multipleOfPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }
} // namespace alpaka::tune
#endif // PARTITIONING_H
