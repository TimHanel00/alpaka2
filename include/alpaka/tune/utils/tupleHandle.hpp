//
// Created by tim on 04.03.25.
//

#ifndef TUPLEHANDLE_H
#define TUPLEHANDLE_H
#include "alpaka/KernelBundle.hpp"
#include "alpaka/tune/active/activeKernel.hpp"
#include "alpaka/tune/active/tuneable.hpp"

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

template<typename T, auto N, typename policy>
struct is_tuneable<alpaka::tune::Tuneable<T, N, policy>> : std::true_type
{
};

template<typename T>
constexpr bool is_tuneable_v = is_tuneable<T>::value;

template<typename>
struct tuneable_underlying
{
};

template<auto N, typename T, typename policy>
struct tuneable_underlying<alpaka::tune::Tuneable<T, N, policy>>
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
    auto valList = tune.makeList();
    return std::make_tuple(
        alpaka::tune::TuneableHandle<ElementType>(
            tune.value[I],
            std::string(tune.name()) + "_" + std::to_string(I),
            tune.userDef,
            tune.idxRange.m_begin[I],
            tune.idxRange.m_end[I],
            tune.idxRange.m_stride[I])...);
}

template<typename Tuneable_type>
auto flatten(Tuneable_type& tune)
{
    constexpr auto dim = alpaka::getDim(typename Tuneable_type::ValueType{});

    // if constexpr(std::is_same_v(Tuneable_type::dimensionTraversePolicy_type, alpaka::tune::DimensionsIndependent))
    //{
    // }
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
    return std::tuple_cat((
        [&]<std::size_t I>(std::integral_constant<std::size_t, I>)
        {
            using ElemType
                = std::decay_t<std::tuple_element_t<I, typename alpaka::KernelBundle<TKernelFn, TArgs...>::ArgTuple>>;

            if constexpr(is_tuneable_v<ElemType>)
            {
                return std::make_tuple(std::get<I>(kb.m_args));
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

// Return empty tuple for NoTune
inline auto makeNonOwningTuneableTuple(alpaka::tune::NoTune const&)
{
    return std::tuple<>{};
}

// Main handler for general tunables
template<typename T_Vec, auto ID>
auto makeNonOwningTuneableTuple(alpaka::tune::Tuneable<T_Vec, ID, alpaka::tune::DimensionsIndependent> const& t)
{
    using elementType = ALPAKA_TYPEOF(t.value);
    using TuneableType = alpaka::tune::Tuneable<T_Vec, ID, alpaka::tune::DimensionsIndependent>;
    if constexpr(alpaka::isVector_v<elementType>)
    {
        // flatten() needs non-const access — cast safely
        return flatten(const_cast<std::remove_cv_t<TuneableType>&>(t));
    }
    else
    {
        static_assert(!sizeof(TuneableType), "Unhandled Tuneable Type does not contain a Vector !");
    }
}

template<typename T_Vec, auto ID>
auto makeNonOwningTuneableTuple(alpaka::tune::Tuneable<T_Vec, ID, alpaka::tune::DimensionsDependent> const& t)
{
    using elementType = ALPAKA_TYPEOF(t.value);
    using TuneableType = alpaka::tune::Tuneable<T_Vec, ID, alpaka::tune::DimensionsDependent>;
    auto& newTune = const_cast<std::remove_cv_t<TuneableType>&>(t);
    auto valList = newTune.makeList();
    if constexpr(alpaka::isVector_v<elementType>)
    {
        // flatten() needs non-const access — cast safely
        return std::make_tuple(
            alpaka::tune::TuneableHandle<elementType>(
                newTune.value,
                std::string(newTune.name()),
                newTune.userDef,
                newTune.idxRange.m_begin,
                newTune.idxRange.m_end,
                newTune.idxRange.m_stride));
    }
    else
    {
        static_assert(!sizeof(TuneableType), "Unhandled Tuneable Type does not contain a Vector !");
    }
}

template<typename Tuple, std::size_t... Is>
auto makeNonOwningframeSpecTuple(Tuple const& tuple, std::index_sequence<Is...>)
{
    return std::tuple_cat(makeNonOwningTuneableTuple(std::get<Is>(tuple))...);
}

template<typename Tuple>
auto makeNonOwningframeSpecTuple(Tuple const& tuple)
{
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    return makeNonOwningframeSpecTuple(tuple, std::make_index_sequence<N>{});
}

template<typename T_KernelRun>
auto makeSharedParameterInterface(T_KernelRun& run)
{
    auto userDef_tuneableTupleInterface = std::apply(
        [](auto&... elems) { return std::tuple_cat(makeNonOwningTuneableTuple(elems)...); },
        run.userTuneables);
    auto CtuneableInterface = std::apply(
        [](auto&... elems) { return std::tuple_cat(makeNonOwningTuneableTuple(elems)...); },
        run.m_compileTimeTuple);
    auto ret = std::tuple_cat(
        userDef_tuneableTupleInterface,
        makeNonOwningframeSpecTuple(run.frameTuneables),
        CtuneableInterface);

    return ret;
}
#endif // TUPLEHANDLE_H
