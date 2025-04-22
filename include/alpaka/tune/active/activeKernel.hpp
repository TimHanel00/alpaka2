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
// concretely defined run for a tuning session stores extracted
/*
 * this object defines the "core" state of the tuning mechanism, defining type-safe tuning parameters in the form of
 *tuples each subsequent parameter configuration is derived from this active State by working in references of its
 *types ..
 **/
template<typename... T_Tuneables>
struct ActiveKernelRun
{
    using T_floating = double_t;
    using T_TuneTuple = std::tuple<T_Tuneables...>;

    T_TuneTuple allTuneables;
    T_floating metric{};
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    strategyState m_strategyState{};
    bool resetSignal{false};

    constexpr ActiveKernelRun() = default;

    explicit ActiveKernelRun(T_Tuneables... tuneables)
        : allTuneables(std::move(tuneables)...)
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
    {
        std::apply(
            [&](auto&... t)
            {
                ((maxRunsDefault *= t.numSteps()), ...);
                maxRuns = maxRunsDefault;
            },
            allTuneables);
    }

    template<alpaka::tune::StaticString Name>
    static constexpr bool hasTuneable()
    {
        return ((T_Tuneables::name() == Name.name()) || ...);
    }

    template<alpaka::tune::StaticString Name>
    auto* getByName()
    {
        return getByNameImpl<Name>(allTuneables);
    }

    template<alpaka::tune::StaticString Name>
    auto const* getByName() const
    {
        return getByNameImpl<Name>(allTuneables);
    }

    template<typename Tuple, alpaka::tune::StaticString Name>
    static constexpr auto* getByNameImpl(Tuple& tuple)
    {
        using RetType = std::common_type_t<std::remove_reference_t<decltype(std::get<0>(tuple))>>;
        RetType* result = nullptr;
        std::apply([&](auto&... elems) { ((elems.name() == Name.name() ? result = &elems : void()), ...); }, tuple);
        return result;
    }

    std::string toHash() const
    {
        std::string hash;
        std::apply([&](auto const&... t) { ((hash += t.toHash()), ...); }, allTuneables);
        return hash;
    }
};

//--------------------------------------
// Factory helper
//--------------------------------------
template<typename... T_Tuneables>
constexpr auto makeActiveKernel(T_Tuneables... tuneables)
{
    return ActiveKernelRun<T_Tuneables...>{std::move(tuneables)...};
}

//--------------------------------------
// Append tuneable if not present
//--------------------------------------
template<typename ExistingKernel, typename NewTuning>
auto appendTuning(ExistingKernel const& kernel, NewTuning const& newTuning)
{
    constexpr auto name = NewTuning::name();

    if constexpr(ExistingKernel::template hasTuneable<NewTuning::name()>())
    {
        std::cout << "TUNER ERROR: tuning already assigned. Skipping.\n";
        return kernel;
    }
    else
    {
        return std::apply(
            [&](auto const&... existing) { return makeActiveKernel(existing..., newTuning); },
            kernel.allTuneables);
    }
}
#endif // ACTIVEKERNEL_H
