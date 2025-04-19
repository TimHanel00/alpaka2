//
// Created by tim on 06.04.25.
//

#ifndef ACTIVEKERNEL_H
#define ACTIVEKERNEL_H
#include "alpaka/tune/active/tuneable.hpp"

#include <cmath>
#include <optional>

struct strategyState
{
    double_t temperature{};
    std::size_t runs{1};
    std::size_t configStamp{0};
    std::string oldKernelHash;
};

template<typename Tuple, std::size_t... I>
static Tuple createDefaultTupleImpl(std::index_sequence<I...>)
{
    return Tuple{std::tuple_element_t<I, Tuple>{}...};
}

template<typename Tuple>
static Tuple createDefaultTuple()
{
    constexpr std::size_t tuple_size = std::tuple_size_v<Tuple>;
    return createDefaultTupleImpl<Tuple>(std::make_index_sequence<tuple_size>{});
}

template<typename... Ts>
struct first_non_void_or
{
    using type = alpaka::tune::NoTune;
};

template<typename T, typename... Ts>
struct first_non_void_or<T, Ts...>
{
    using type = std::conditional_t<std::is_same_v<T, void>, typename first_non_void_or<Ts...>::type, T>;
};

template<typename... Ts>
using type_or = typename first_non_void_or<Ts...>::type;

template<typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template<template<typename> class Predicate, typename... Ts>
struct find_type_if
{
    using type = type_or<std::conditional_t<Predicate<remove_cvref_t<Ts>>::value, remove_cvref_t<Ts>, void>...>;
};

template<template<typename> class Predicate, typename... Ts>
using find_type_if_t = typename find_type_if<Predicate, Ts...>::type;

template<typename T>
struct is_NumBlocks : std::false_type
{
    using type = alpaka::tune::NoTune;
};

template<typename T>
struct is_NumBlocks<alpaka::tune::NumBlocksTune<T>> : std::true_type
{
    using type = alpaka::tune::NumBlocksTune<T>;
};

template<typename T>
struct is_ThreadBlockSize : std::false_type
{
    using type = alpaka::tune::NoTune;
};

template<typename T>
struct is_ThreadBlockSize<alpaka::tune::ThreadBlockSizeTune<T>> : std::true_type
{
    using type = alpaka::tune::ThreadBlockSizeTune<T>;
};

template<typename T>
struct is_NumFrames : std::false_type
{
    using type = alpaka::tune::NoTune;
};

template<typename T>
struct is_NumFrames<alpaka::tune::NumFramesTune<T>> : std::true_type
{
    using type = alpaka::tune::NumFramesTune<T>;
};

template<typename T>
struct is_FrameExtent : std::false_type
{
    using type = alpaka::tune::NoTune;
};

template<typename T>
struct is_FrameExtent<alpaka::tune::FrameExtentTune<T>> : std::true_type
{
    using type = alpaka::tune::FrameExtentTune<T>;
};

static constexpr alpaka::tune::NoTune noTune{};

template<typename T, typename Tuple, std::size_t... Is>
constexpr bool frameTunecontainsImpl(std::index_sequence<Is...>)
{
    return (
        ...
        || std::is_same_v<
            std::remove_cv_t<std::remove_reference_t<T>>,
            std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<Is, Tuple>>>>);
}

template<typename T, typename Tuple>
constexpr bool frameTunecontains()
{
    using T_raw = std::remove_cv_t<std::remove_reference_t<T>>;
    using NoTune_t = alpaka::tune::NoTune;

    if constexpr(std::is_same_v<T_raw, NoTune_t>)
    {
        return false;
    }
    else
    {
        return frameTunecontainsImpl<T, Tuple>(
            std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
    }
}

template<typename T_Target, typename Tuple, std::size_t I = 0>
constexpr auto& get_from_tuple_impl(Tuple& tuple)
{
    using T_clean = std::remove_cvref_t<T_Target>;
    using TupleT = std::remove_cvref_t<Tuple>;
    if constexpr(I >= std::tuple_size_v<TupleT>)
    {
        return noTune;
    }
    else if constexpr(std::is_same_v<T_clean, std::remove_cvref_t<std::tuple_element_t<I, TupleT>>>)
    {
        return std::get<I>(tuple); // ✅ returns lvalue-ref
    }
    else
    {
        return get_from_tuple_impl<T_Target, Tuple, I + 1>(tuple); // ✅ recurses with ref
    }
}

template<typename T_Target, typename Tuple>
constexpr auto& get_from_tuple(Tuple& tuple)
{
    using T_clean = std::remove_cvref_t<T_Target>;
    if constexpr(std::is_same_v<T_clean, alpaka::tune::NoTune>)
    {
        return noTune; // must also be reference if you're taking its address
    }
    else
    {
        return get_from_tuple_impl<T_clean>(tuple);
    }
}

// concretely defined run for a tuning session stores extracted
template<
    typename T_numFramesTune = alpaka::tune::NoTune,
    typename T_frameExtentTune = alpaka::tune::NoTune,
    typename T_numBlocksTune = alpaka::tune::NoTune,
    typename T_numThreadsTune = alpaka::tune::NoTune,
    typename T_UserDefTuneablesTune = std::tuple<>>
struct ActiveKernelRun
{
    using T_floating = double_t;
    using T_frameSpecTuple = std::tuple<T_numFramesTune, T_frameExtentTune, T_numBlocksTune, T_numThreadsTune>;
    T_frameSpecTuple frameSpecTuple;
    T_floating metric{};
    T_UserDefTuneablesTune userDefTuneables;
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    strategyState m_strategyState{};
    bool resetSignal{false};
    constexpr ActiveKernelRun() = default;

    template<typename T_FrameTunings>
    constexpr explicit ActiveKernelRun(T_UserDefTuneablesTune const& args, T_FrameTunings frameTune)
        : frameSpecTuple(std::move(frameTune))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
        , userDefTuneables(args)
    {
        for_each(
            userDefTuneables,
            [&](auto& argsT)
            {
                std::size_t maxRunsDefault_old = maxRunsDefault;
                maxRunsDefault *= argsT.numSteps();
                if(maxRunsDefault_old > maxRunsDefault)
                {
                    std::cout << " WARNING: Overflow detected during tuning space calculation, ensure "
                                 "you have a max NumofRuns selected!"
                              << std::endl;
                    maxRunsDefault = UINT64_MAX;
                }
            });
        maxRuns = maxRunsDefault;
    };

    static constexpr bool hasNumBlocksTune()
    {
        return frameTunecontains<T_numBlocksTune, T_frameSpecTuple>();
    }

    static constexpr bool hasThreadBlockSizeTune()
    {
        return frameTunecontains<T_numThreadsTune, T_frameSpecTuple>();
    }

    static constexpr bool hasNumFramesTune()
    {
        return frameTunecontains<T_numFramesTune, T_frameSpecTuple>();
    }

    static constexpr bool hasFrameExtentTune()
    {
        return frameTunecontains<T_frameExtentTune, T_frameSpecTuple>();
    }

    auto& getNumBlocksTune()
    {
        return get_from_tuple<T_numBlocksTune>(frameSpecTuple);
    }

    auto& getThreadBlockSizeTune()
    {
        return get_from_tuple<T_numThreadsTune>(frameSpecTuple);
    }

    auto& getNumFramesTune()
    {
        return get_from_tuple<T_numFramesTune>(frameSpecTuple);
    }

    auto& getFrameExtentTune()
    {
        return get_from_tuple<T_frameExtentTune>(frameSpecTuple);
    }

    auto const& getNumBlocksTune() const
    {
        return get_from_tuple<T_numBlocksTune>(frameSpecTuple);
    }

    auto const& getThreadBlockSizeTune() const
    {
        return get_from_tuple<T_numThreadsTune>(frameSpecTuple);
    }

    auto const& getNumFramesTune() const
    {
        return get_from_tuple<T_numFramesTune>(frameSpecTuple);
    }

    auto const& getFrameExtentTune() const
    {
        return get_from_tuple<T_frameExtentTune>(frameSpecTuple);
    }

    std::string toHash()
    {
        std::string m;
        std::apply(
            [&m](auto const&... args)
            {
                ((m += args.toHash()), ...); // fold expression over the comma operator for user defined Tuneables
            },
            userDefTuneables);
        m += getNumFramesTune().toHash();
        m += getFrameExtentTune().toHash();
        m += getNumBlocksTune().toHash();
        m += getThreadBlockSizeTune().toHash();
        return m;
        // return "";
    }
};

template<typename T_UserDefTuneables, typename... T_tunings>
auto makeActiveKernel(T_UserDefTuneables const& defs, T_tunings&&... tunings)
{
    using T_numFrames = find_type_if_t<is_NumFrames, T_tunings...>;
    using T_frameExtent = find_type_if_t<is_FrameExtent, T_tunings...>;
    using T_numBlocks = find_type_if_t<is_NumBlocks, T_tunings...>;
    using T_numThreads = find_type_if_t<is_ThreadBlockSize, T_tunings...>;
    auto tup = std::forward_as_tuple(std::forward<T_tunings>(tunings)...);
    auto frameTune = get_from_tuple<T_numFrames>(tup);
    auto frameExtentTune = get_from_tuple<T_frameExtent>(tup);
    auto numBlocksTune = get_from_tuple<T_numBlocks>(tup);
    auto numThreadsTune = get_from_tuple<T_numThreads>(tup);
    auto frameSpecTuple = std::tuple<
        std::remove_cvref_t<decltype(frameTune)>,
        std::remove_cvref_t<decltype(frameExtentTune)>,
        std::remove_cvref_t<decltype(numBlocksTune)>,
        std::remove_cvref_t<decltype(numThreadsTune)>>{frameTune, frameExtentTune, numBlocksTune, numThreadsTune};

    auto kernel = ActiveKernelRun<T_numFrames, T_frameExtent, T_numBlocks, T_numThreads, T_UserDefTuneables>{
        defs,
        frameSpecTuple};
    return kernel;
}

template<typename T_Tune, typename T_Kernel>
constexpr bool containsTunable()
{
    if constexpr(T_Kernel::hasNumFramesTune() && is_NumFrames<T_Tune>{})
    {
        return true;
    }
    if constexpr(T_Kernel::hasFrameExtentTune() && is_FrameExtent<T_Tune>{})
    {
        return true;
    }
    if constexpr(T_Kernel::hasNumBlocksTune() && is_NumBlocks<T_Tune>{})
    {
        return true;
    }
    if constexpr(T_Kernel::hasThreadBlockSizeTune() && is_ThreadBlockSize<T_Tune>{})
    {
        return true;
    }
    return false;
}

template<typename ExistingKernel, typename NewTuning>
auto appendTuning(ExistingKernel const& kernel, NewTuning const& newTuning)
{
    if constexpr(containsTunable<NewTuning, ExistingKernel>())
    {
        std::cout << " TUNER ERROR: this tuning was already assigned to this builder/session .. skipping assignment"
                  << std::endl;


        return kernel;
    }
    else
    {
        auto tuningsTuple = std::tuple_cat(kernel.frameSpecTuple, std::make_tuple(newTuning));
        return std::apply(
            [&]<typename... T0>(T0&&... args)
            { return makeActiveKernel(kernel.userDefTuneables, std::forward<T0>(args)...); },
            tuningsTuple);
    }
}
#endif // ACTIVEKERNEL_H
