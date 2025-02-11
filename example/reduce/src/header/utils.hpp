//
// Created by tim on 03.01.25.
//

#ifndef UTILS_H
#define UTILS_H
#include "commonIncludes.hpp"

template<typename T>
class GenericUniformDistribution {
public:
    GenericUniformDistribution(T a, T b) : dist(a, b) {}

    template<typename RNG>
    T operator()(RNG& gen) {
        return dist(gen);
    }

private:
    typename std::conditional<std::is_integral<T>::value,
                              std::uniform_int_distribution<T>,
                              std::uniform_real_distribution<T>>::type dist;
};
#endif //UTILS_H
