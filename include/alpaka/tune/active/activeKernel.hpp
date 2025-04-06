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
    std::size_t runs{0};
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

// concretely defined run for a tuning session stores extracted
template<typename T_GridSizeTune, typename T_BlockSizeTune, typename T_TuneableType>
struct ActiveKernelRun
{
    using T_floating = double_t;
    T_GridSizeTune gridSize;
    T_BlockSizeTune threadBlockSize;
    T_floating metric{};
    T_TuneableType tuneables;
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    strategyState m_strategyState{};
    bool resetSignal{false};
    constexpr ActiveKernelRun() = default;

    constexpr ActiveKernelRun(
        T_GridSizeTune const& gridSize,
        T_BlockSizeTune const& blockSize,
        T_TuneableType const& args)
        : gridSize(std::move(gridSize))
        , threadBlockSize(std::move(blockSize))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
        , tuneables(args)
    {
        for_each(
            tuneables,
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

    std::string toHash()
    {
        std::string m;
        std::apply(
            [&m](auto const&... args)
            {
                ((m += args.toHash()), ...); // fold expression over the comma operator
            },
            tuneables);
        if(gridSize.has_value())
            m += gridSize->toHash();
        if(threadBlockSize.has_value())
            m += threadBlockSize->toHash();
        return m;
        // return "";
    }

    auto createActiveDummyKernel() const
    {
        // Default-constructed tuple
        T_TuneableType defaultTuneables = createDefaultTuple<T_TuneableType>();

        return std::move(
            ActiveKernelRun<T_GridSizeTune, T_BlockSizeTune, T_TuneableType>(
                std::nullopt,
                std::nullopt,
                defaultTuneables));
    }
};

#endif // ACTIVEKERNEL_H
