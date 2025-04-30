//
// Created by tim on 06.04.25.
//

#ifndef ACTIVEKERNEL_H
#define ACTIVEKERNEL_H
#include "alpaka/tune/active/tuneable.hpp"

#include <alpaka/tune/utils/tupleHandle.hpp>

#include <cmath>

struct strategyState
{
    double_t temperature{};
    bool done = false;
    std::size_t runs{1};
    std::size_t configStamp{0};
    std::string oldKernelHash;
};

template<typename T>
inline constexpr bool is_empty_tuple_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::tuple<>>;

// concretely defined run for a tuning session stores extracted
/*
 * this object defines the "core" state of the tuning mechanism, defining type-safe tuning parameters in the form of
 *tuples each subsequent parameter configuration is derived from this active State by working in references of its
 *types ..
 **/
template<typename T_UserTuple = std::tuple<>, typename T_FrameTuneables = std::tuple<>>
struct ActiveKernelRun
{
    using T_floating = double_t;
    using T_TuneTuple = decltype(std::tuple_cat(std::declval<T_FrameTuneables>(), std::declval<T_UserTuple>()));
    T_UserTuple userTuneables;
    T_FrameTuneables frameTuneables;


    T_floating metric{};
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    strategyState m_strategyState{};
    bool resetSignal{false};

    constexpr ActiveKernelRun() = default;

    constexpr explicit ActiveKernelRun(T_UserTuple userT, T_FrameTuneables frameT)
        : userTuneables(std::move(userT))
        , frameTuneables(std::move(frameT))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
    {
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, userTuneables);
        maxRuns = maxRunsDefault;
    }

    template<typename Tuple, auto Name, std::size_t I = 0>
    static constexpr bool hasTagInTuple()
    {
        if constexpr(is_empty_tuple_v<Tuple>)
        {
            return false;
        }
        else if constexpr(I < std::tuple_size_v<Tuple>)
        {
            using Elem = std::tuple_element_t<I, Tuple>;
            return (Elem::tag == Name) || hasTagInTuple<Tuple, Name, I + 1>();
        }
        else
        {
            return false;
        }
    }

    constexpr auto allTuneables() &
    {
        return std::apply(
            [this](auto&... frameElems)
            {
                return std::apply(
                    [&](auto&... userElems) { return std::tie(frameElems..., userElems...); },
                    userTuneables);
            },
            frameTuneables);
    }

    constexpr auto allTuneables() const&
    {
        return std::apply(
            [this](auto const&... frameElems)
            {
                return std::apply(
                    [&](auto const&... userElems) { return std::tie(frameElems..., userElems...); },
                    userTuneables);
            },
            frameTuneables);
    }

    template<typename Tuple, auto ID>
    static constexpr bool hasTuneableTag()
    {
        return hasTagInTuple<Tuple, ID>();
    }

    template<auto ID>
    static constexpr bool hasFrameTuneable()
    {
        constexpr auto ID_v = static_cast<std::size_t>(ID);
        return hasTuneableTag<T_FrameTuneables, ID_v>();
    }

    template<auto ID>
    static constexpr bool hasUserTuneable()
    {
        constexpr auto ID_v = static_cast<std::size_t>(ID);
        return hasTuneableTag<T_UserTuple, ID_v>();
    }

    template<auto ID>
    static constexpr bool hasTuneable()
    {
        return hasTuneableTag<T_FrameTuneables, ID>() || hasTuneableTag<T_UserTuple, ID>();
    }

    static constexpr bool hasNumBlocksTune()
    {
        return hasFrameTuneable<alpaka::tune::SpecialTuneableID::NumBlocks>();
    }

    static constexpr bool hasThreadBlockSizeTune()
    {
        return hasFrameTuneable<alpaka::tune::SpecialTuneableID::ThreadBlock>();
    }

    static constexpr bool hasNumFramesTune()
    {
        return hasFrameTuneable<alpaka::tune::SpecialTuneableID::NumFrames>();
    }

    static constexpr bool hasFrameExtentTune()
    {
        return hasFrameTuneable<alpaka::tune::SpecialTuneableID::FrameExtent>();
    }

    constexpr auto& getNumBlocksTune()
    {
        return *getByID<alpaka::tune::SpecialTuneableID::NumBlocks>();
    }

    constexpr auto& getThreadBlockSizeTune()
    {
        return *getByID<alpaka::tune::SpecialTuneableID::ThreadBlock>();
    }

    constexpr auto& getNumFramesTune()
    {
        return *getByID<alpaka::tune::SpecialTuneableID::NumFrames>();
    }

    constexpr auto& getFrameExtentTune()
    {
        return *getByID<alpaka::tune::SpecialTuneableID::FrameExtent>();
    }

    template<auto Name>
    constexpr auto& getValue()
    {
        return getByID<Name>()->value;
    }

    template<auto Name>
    constexpr auto const& getValue() const
    {
        return getByID<Name>()->value;
    }

    template<std::size_t I = 0, typename Tuple, auto ID>
    constexpr auto* getByIDImpl(Tuple& tuple) const
    {
        if constexpr(I < std::tuple_size_v<Tuple>)
        {
            auto& elem = std::get<I>(tuple);
            using elemType = std::decay_t<decltype(elem)>;
            if constexpr(elemType::tag == ID)
            {
                return &elem;
            }
            else
            {
                return getByIDImpl<I + 1, Tuple, ID>(tuple);
            }
        }
        else
        {
            return &alpaka::tune::noTune; // or nullptr if appropriate
        }
    }

    template<auto ID>
    constexpr auto* getByID()
    {
        constexpr auto ID_v = static_cast<std::size_t>(ID);
        auto all = allTuneables();
        return getByIDImpl<0, decltype(all), ID_v>(all);
    }

    std::string toHash() const
    {
        std::string hash;
        std::apply([&](auto const&... t) { ((hash += t.toHash()), ...); }, allTuneables());
        return hash;
    }
};

//--------------------------------------
// Factory helper
//--------------------------------------
template<typename T>
constexpr bool is_not_no_tune_v = !std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, alpaka::tune::NoTune>;

template<typename UserTuple, typename... FrameT>
constexpr auto makeActiveKernel(UserTuple userT, FrameT&&... frameArgs)
{
    // Filter FrameT pack by excluding alpaka::tune::NoTune
    auto filteredTuple = std::tuple_cat(
        []<typename T>(T&& arg) -> auto
        {
            if constexpr(!std::is_same_v<std::decay_t<T>, alpaka::tune::NoTune>)
                return std::make_tuple(std::forward<T>(arg));
            else
                return std::tuple<>{}; // drop NoTune
        }(std::forward<FrameT>(frameArgs))... // expand pack
    );
    using FilteredTuple = decltype(filteredTuple);
    return ActiveKernelRun<UserTuple, FilteredTuple>{std::move(userT), std::move(filteredTuple)};
}

//--------------------------------------
// Append tuneable if not present
//--------------------------------------
template<typename ExistingKernel, typename NewTuning>
auto appendTuning(ExistingKernel const& kernel, NewTuning const& newTuning)
{
    if constexpr(ExistingKernel::template hasFrameTuneable<newTuning.tag>())
    {
        std::cout << "TUNER ERROR: tuning already assigned. Skipping.\n";
        return kernel;
    }
    else if constexpr(std::is_same_v<alpaka::tune::NoTune, std::remove_cvref_t<NewTuning>>)
    {
        return kernel;
    }
    else
    {
        auto newFrameTuple = std::apply(
            [&](auto const&... elems) { return std::make_tuple(elems..., newTuning); },
            kernel.frameTuneables);

        return ActiveKernelRun{kernel.userTuneables, std::move(newFrameTuple)};
    }
}

#endif // ACTIVEKERNEL_H
