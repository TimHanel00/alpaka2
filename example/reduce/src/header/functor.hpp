//
// Created by tim on 03.01.25.
//

#ifndef FUNCTOR_H
#define FUNCTOR_H
#include "commonIncludes.hpp"
namespace reduce {
        struct sum { template<typename T> constexpr T operator()(T const& a, T const& b) const { return a + b; } };
        struct multiply { template<typename T> constexpr T operator()(T const& a, T const& b) const { return a * b; } };
        struct max_element { template<typename T> T operator()(T const& a, T const& b) const { return (a > b) ? a : b; } };
        struct min_element { template<typename T> T operator()(T const& a, T const& b) const { return (a < b) ? a : b; } };
        struct all { template<typename T> T operator()(T const& a, T const& b) const { return a && b; } };
        struct any { template<typename T> T operator()(T const& a, T const& b) const { return a || b; } };
        struct xor_ { template<typename T> T operator()(T const& a, T const& b) const { return a ^ b; } };
        struct bitwise_and { template<typename T> T operator()(T const& a, T const& b) const { return a & b; } };
        struct bitwise_or { template<typename T> T operator()(T const& a, T const& b) const { return a | b; } };

// Generic operate function: Works with functors and lambdas
template <typename Op, typename T>
constexpr auto operate(Op&& operation, T const& op1, T const& op2) -> T {
    return std::forward<Op>(operation)(op1, op2);
}
    template <typename T, typename OperationType>
ALPAKA_FN_HOST_ACC auto neutral_element(const OperationType& type) -> T
    {
        if constexpr (std::is_same_v<OperationType, reduce::sum>) {
            return T(0); // Neutral element for addition
        } else if constexpr (std::is_same_v<OperationType, reduce::multiply>) {
            return T(1); // Neutral element for multiplication
        } else if constexpr (std::is_same_v<OperationType, reduce::max_element>) {
            return std::numeric_limits<T>::lowest(); // Neutral element for max (smallest possible value)
        } else if constexpr (std::is_same_v<OperationType, reduce::min_element>) {
            return std::numeric_limits<T>::max(); // Neutral element for min (largest possible value)
        } else if constexpr (std::is_same_v<OperationType, reduce::all>) {
            return true; // Neutral element for logical AND (all true)
        } else if constexpr (std::is_same_v<OperationType, reduce::any>) {
            return false; // Neutral element for logical OR (any true)
        } else if constexpr (std::is_same_v<OperationType, reduce::xor_>) {
            return T(0); // Neutral element for XOR
        } else if constexpr (std::is_same_v<OperationType, reduce::bitwise_and>) {
            return ~T(0); // Neutral element for bitwise AND (all bits set)
        } else if constexpr (std::is_same_v<OperationType, reduce::bitwise_or>) {
            return T(0); // Neutral element for bitwise OR
        } else {
            return T(0);
            //static_assert(std::is_same_v<OperationType, void>, "Unsupported operation");
        }
    }
        template <typename Acc,typename OperationType,typename T>
    ALPAKA_FN_HOST_ACC auto atomicOp(Acc const &acc,const OperationType& type,auto destPtr,T elem) -> void {
                if constexpr (std::is_same_v<OperationType, reduce::sum>) {
                    alpaka::onAcc::atomicAdd(acc,destPtr,elem); // atomic for add
                } else if constexpr (std::is_same_v<OperationType, reduce::multiply>) {
                    static_assert(std::is_same_v<OperationType, void>, "Atomic Multiply currently not supported");
                } else if constexpr (std::is_same_v<OperationType, reduce::max_element>) {
                    alpaka::onAcc::atomicMax(acc,destPtr,elem);
                } else if constexpr (std::is_same_v<OperationType, reduce::min_element>) {
                    alpaka::onAcc::atomicMin(acc,destPtr,elem);
                } else if constexpr (std::is_same_v<OperationType, reduce::all>) {
                    alpaka::onAcc::atomicAnd(acc,destPtr,elem);
                } else if constexpr (std::is_same_v<OperationType, reduce::any>) {
                    alpaka::onAcc::atomicOr(acc,destPtr,elem);
                } else {
                    static_assert(std::is_same_v<OperationType, void>, "Unsupported atomic operation");
                }
    }
            // Neutral element for lambda operation
// Operation class template


}
#endif //FUNCTOR_H
