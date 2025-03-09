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
// 1) Primary template
template<typename X>
struct is_tuneable : std::false_type {};

// 2) Partial specialization for any instantiation of alpaka::tune::Tuneable<...>
template<typename T, typename T_End, typename T_Begin, typename T_Stride>
struct is_tuneable< alpaka::tune::Tuneable<T, T_End, T_Begin, T_Stride> >
    : std::true_type {};

// 3) The variable template convenience alias
template<typename X>
constexpr bool is_tuneable_v = is_tuneable<X>::value;
template<typename Arg,typename T>
void addIfTuneable(
    Arg& arg,
    std::vector<std::shared_ptr<alpaka::tune::Tuneable<T>>>& result)
{
    if constexpr (is_tuneable_v<Arg>) {
        // Create a shared_ptr that does NOT own 'arg'
        // We supply a no-op deleter to avoid double-free
        auto ptr = std::shared_ptr<Arg>(&arg, [](Arg*){ /* no-op */ });
        result.push_back(ptr);
    }
}

template<typename T,typename TKernelFn, typename... TArgs>
auto extractTuneables(const alpaka::KernelBundle<TKernelFn, TArgs...>& kb)
    -> std::vector<std::shared_ptr<alpaka::tune::Tuneable<T>>>
{
    std::vector<std::shared_ptr<alpaka::tune::Tuneable<T>>> result;

    // Access the underlying tuple
    auto& argsTuple = kb.m_args;

    // For each element in 'argsTuple', check if it's a tuneable.
    std::apply(
        [&](auto&... elems) {
            (addIfTuneable(elems, result), ...);
        },
        argsTuple
    );
    return result;
}
#endif //TUPLEHANDLE_H
