//
// Created by tim on 04.03.25.
//

#ifndef TUPLEHANDLE_H
#define TUPLEHANDLE_H
#include <string>
#include <vector>

inline void processArgs(std::vector<std::string>&) {}

// Recursive case: contains arithmetric Argument
template <typename First, typename... Rest>
void processArgs(std::vector<std::string>& specifierStrings, First first, Rest... rest) {
    if constexpr (std::is_arithmetic_v<First>) {
        specifierStrings.push_back(std::to_string(first)); // Convert numbers to strings
    }
    processArgs(specifierStrings, rest...); // Process remaining args
}

// Recursive case: contains string Argument
template<typename... Args>
void processArgs(std::vector<std::string>& specifierStrings, const std::string& str, Args... rest) {
    specifierStrings.push_back(str);
    processArgs(specifierStrings, rest...);
}
template<typename>
struct is_tuneable : std::false_type {};

template<typename T, typename T_End, typename T_Begin, typename T_Stride>
struct is_tuneable<alpaka::tune::Tuneable<T, T_End, T_Begin, T_Stride>> : std::true_type {};

template<typename T>
constexpr bool is_tuneable_v = is_tuneable<T>::value;

template<typename>
struct tuneable_underlying {};

template<typename T, typename T_End, typename T_Begin, typename T_Stride>
struct tuneable_underlying<alpaka::tune::Tuneable<T, T_End, T_Begin, T_Stride>> {
    using type = T;
};
// Helper: transform a tuple by applying a callable that receives an index and the element.
template <typename Tuple, typename F, std::size_t... Is>
auto tuple_transform_with_index(Tuple&& tup, F f, std::index_sequence<Is...>)
{
    return std::make_tuple(f(std::integral_constant<std::size_t, Is>{}, std::get<Is>(std::forward<Tuple>(tup)))...);
}

template<typename T, typename TKernelFn, typename... TArgs>
auto recreate(const alpaka::KernelBundle<TKernelFn, TArgs...>& kb,
              const std::vector<alpaka::tune::Tuneable<T>>& newTuneables)
{
    size_t tune_idx = 0;
    auto new_args = tuple_transform_with_index(kb.m_args,
        [&]<typename I, typename Elem>(I /*index*/, const Elem& elem) -> auto {
            using ElemType = std::decay_t<Elem>;
            if constexpr (is_tuneable_v<ElemType>) {
                // Replace the tuneable with its 'value' (assume order matches).
                return newTuneables[tune_idx++].value;
            } else {
                return elem;
            }
        },
        std::make_index_sequence<sizeof...(TArgs)>{});

    // Reconstruct a new KernelBundle from the same kernel function and the new tuple.
    // We use std::move on the elements to ensure they bind to the rvalue reference parameters.
    return std::apply(
        [&]<typename... U>(U&&... elems) {
            return alpaka::KernelBundle<TKernelFn, std::decay_t<U>...>(
                kb.m_kernelFn, std::move(elems)...);
        },
        new_args);
}

// 1) Primary template



template<typename T, typename TKernelFn, typename... TArgs, std::size_t... Is>
void extractTuneables_impl(
    const alpaka::KernelBundle<TKernelFn, TArgs...>& kb,
    std::vector<alpaka::tune::Tuneable<T>>& vec,
    std::index_sequence<Is...>)
{
    (void)std::initializer_list<int>{
        ( [&]() -> int {
             using ElemType = std::decay_t<std::tuple_element_t<Is, typename alpaka::KernelBundle<TKernelFn, TArgs...>::ArgTuple>>;
             if constexpr (is_tuneable_v<ElemType>) {
                 using Underlying = typename tuneable_underlying<ElemType>::type;
                 static_assert(std::is_same_v<Underlying, T>,
                               "Inconsistent tuneable underlying type encountered!");
                 vec.push_back(std::get<Is>(kb.m_args));
             }
             return 0;
         }() )...
    };
}

template<typename T, typename TKernelFn, typename... TArgs>
std::vector<alpaka::tune::Tuneable<T>> extractTuneables(const alpaka::KernelBundle<TKernelFn, TArgs...>& kb)
{
    std::vector<alpaka::tune::Tuneable<T>> vec;
    extractTuneables_impl(kb, vec, std::make_index_sequence<sizeof...(TArgs)>{});
    return vec;
}
#endif //TUPLEHANDLE_H
