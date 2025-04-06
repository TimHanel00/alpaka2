//
// Created by tim on 04.03.25.
//

#ifndef TUPLEHANDLE_H
#define TUPLEHANDLE_H
#include "alpaka/KernelBundle.hpp"

#include <string>
#include <vector>

inline void processArgs(std::vector<std::string>&)
{
}

// Recursive case: contains arithmetric Argument
template<typename First, typename... Rest>
void processArgs(std::vector<std::string>& specifierStrings, First first, Rest... rest)
{
    if constexpr(std::is_arithmetic_v<First>)
    {
        specifierStrings.push_back(std::to_string(first)); // Convert numbers to strings
    }
    processArgs(specifierStrings, rest...); // Process remaining args
}

// Recursive case: contains string Argument
template<typename... Args>
void processArgs(std::vector<std::string>& specifierStrings, std::string const& str, Args... rest)
{
    specifierStrings.push_back(str);
    processArgs(specifierStrings, rest...);
}

template<typename>
struct is_tuneable : std::false_type
{
};

template<typename T>
struct is_tuneable<alpaka::tune::Tuneable<T>> : std::true_type
{
};

template<typename T>
constexpr bool is_tuneable_v = is_tuneable<T>::value;

template<typename>
struct tuneable_underlying
{
};

template<typename T>
struct tuneable_underlying<alpaka::tune::Tuneable<T>>
{
    using type = T;
};

// Helper: transform a tuple by applying a callable that receives an index and the element.
template<typename Tuple, typename F, std::size_t... Is>
auto tuple_transform_with_index(Tuple&& tup, F f, std::index_sequence<Is...>)
{
    return std::make_tuple(f(std::integral_constant<std::size_t, Is>{}, std::get<Is>(std::forward<Tuple>(tup)))...);
}

template<std::size_t I, typename Tuple>
constexpr std::size_t count_tuneables()
{
    if constexpr(I == 0)
    {
        return 0;
    }
    else
    {
        constexpr bool isPrev
            = is_tuneable_v<std::remove_reference_t<decltype(std::get<I - 1>(std::declval<Tuple>()))>>;
        return count_tuneables<I - 1, Tuple>() + (isPrev ? 1 : 0);
    }
}

template<typename TuneableType, std::size_t... I>
auto flattenImpl(TuneableType& tune, std::index_sequence<I...>)
{
    using VecType = std::remove_reference_t<decltype(tune.value)>;
    using ElementType = typename VecType::type;

    return std::make_tuple(
        alpaka::tune::FlatTuneableHandle<ElementType>(
            tune.value[I],
            tune.name + "_" + std::to_string(I),
            tune.userDef,
            tune.idxRange.m_begin[I],
            tune.idxRange.m_end[I],
            tune.idxRange.m_stride[I])...);
}

template<typename T>
auto flatten(alpaka::tune::Tuneable<T>& tune)
{
    constexpr auto dim = alpaka::getDim(T{});
    return flattenImpl(tune, std::make_index_sequence<dim>{});
}

// Modified recreate: now newTuneables is a tuple of tuneables, each with a .value member.
template<typename TKernelFn, typename... TArgs, typename TTuneableNew>
auto recreate(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb, TTuneableNew&& newTuneables)
{
    // Transform the kernel bundle's arguments: when a tuneable is encountered,
    // substitute its value from newTuneables (using the compile-time computed index).
    auto new_args = tuple_transform_with_index(
        kb.m_args,
        [&]<std::size_t I, typename Elem>(std::integral_constant<std::size_t, I>, Elem const& elem) -> auto
        {
            using ElemType = std::decay_t<Elem>;
            if constexpr(is_tuneable_v<ElemType>)
            {
                constexpr std::size_t tune_index = count_tuneables<I, decltype(kb.m_args)>();
                return std::get<tune_index>(newTuneables).value;
            }
            else
            {
                return elem;
            }
        },
        std::make_index_sequence<std::tuple_size_v<decltype(kb.m_args)>>{});

    // Reconstruct a new KernelBundle from the same kernel function and the new tuple.
    return std::apply(
        [&]<typename... U>(U&&... elems)
        { return alpaka::KernelBundle<TKernelFn, std::decay_t<U>...>(kb.m_kernelFn, std::move(elems)...); },
        new_args);
}

// 1) Primary template


template<typename TKernelFn, typename... TArgs, std::size_t... Is>
auto extractTuneables_impl(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb, std::index_sequence<Is...>)
{
    std::size_t tuneableIdx = 0;

    return std::tuple_cat((
        [&]<std::size_t I>(std::integral_constant<std::size_t, I>)
        {
            using ElemType
                = std::decay_t<std::tuple_element_t<I, typename alpaka::KernelBundle<TKernelFn, TArgs...>::ArgTuple>>;

            if constexpr(is_tuneable_v<ElemType>)
            {
                auto tune = std::get<I>(kb.m_args);
                constexpr std::string_view defaultName = "Tuneable: ";

                std::ostringstream oss;
                oss << tune.name;

                if(tune.name == defaultName)
                    oss << tuneableIdx;

                tune.name = oss.str();

                tuneableIdx++; // increment runtime counter
                return std::make_tuple(tune);
            }
            else
            {
                return std::tuple<>();
            }
        }(std::integral_constant<std::size_t, Is>{}))...);
}

template<typename TKernelFn, typename... TArgs>
auto extractTuneables(alpaka::KernelBundle<TKernelFn, TArgs...> const& kb)
{
    return extractTuneables_impl(kb, std::make_index_sequence<sizeof...(TArgs)>{});
}

template<typename TuneableType>
auto makeNonOwningTuneableTuple(TuneableType& t)
{
    using ValueType = std::remove_reference_t<TuneableType>;
    using elementType = ALPAKA_TYPEOF(t.value);
    if constexpr(alpaka::isVector_v<elementType>)
    {
        return flatten(t);
    }
    else if constexpr(std::is_integral_v<elementType>)
    {
        return std::tuple(
            alpaka::tune::FlatTuneableHandle<elementType>{
                t.value,
                t.name,
                t.userDef,
                t.idxRange.m_begin[0],
                t.idxRange.m_end[0],
                t.idxRange.m_stride[0]});
    }
    throw std::runtime_error("unrecognized tuneableType");
}

template<bool grid, bool block, typename T_ActiveKernel>
auto makeSharedParameterInterface(T_ActiveKernel& run)
{
    auto tuneableTuple = std::apply(
        [](auto&... elems) { return std::tuple_cat(makeNonOwningTuneableTuple(elems)...); },
        run.tuneables);
    if constexpr(grid && block)
    {
        auto gridTuple = makeNonOwningTuneableTuple(run.gridSize);
        auto blockTuple = makeNonOwningTuneableTuple(run.threadBlockSize);
        return std::tuple_cat(tuneableTuple, gridTuple, blockTuple);
    }
    else if constexpr(grid)
    {
        auto gridTuple = makeNonOwningTuneableTuple(run.gridSize);
        return std::tuple_cat(tuneableTuple, gridTuple);
    }
    else if constexpr(block)
    {
        auto blockTuple = makeNonOwningTuneableTuple(run.threadBlockSize);
        return std::tuple_cat(tuneableTuple, blockTuple);
    }
    else
    {
        return tuneableTuple;
    }
}
#endif // TUPLEHANDLE_H
