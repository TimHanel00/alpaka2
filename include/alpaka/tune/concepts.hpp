//
// Created by tim on 08.10.25.
//

#ifndef CONCEPTS_H
#define CONCEPTS_H

namespace alpaka::tune::concepts
{
    template<typename T>
    concept ArithmeticComparable = requires(T a, T b) {
        { a + b } -> std::same_as<T>;
        { a < b } -> std::convertible_to<bool>;
        { a <= b } -> std::convertible_to<bool>;
        { a == b } -> std::convertible_to<bool>;
    };
    template<typename T>
    concept Floating = std::is_floating_point_v<T>;
    template<typename T>
    concept Integral = std::is_integral_v<T>;
    template<typename T>
    concept ArithmeticComparableOrVec = ArithmeticComparable<T> || isVector_v<T>;
} // namespace alpaka::tune::concepts
#endif // CONCEPTS_H
