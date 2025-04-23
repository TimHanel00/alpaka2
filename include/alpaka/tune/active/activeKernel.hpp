//
// Created by tim on 06.04.25.
//

#ifndef ACTIVEKERNEL_H
#define ACTIVEKERNEL_H
#include "alpaka/tune/active/tuneable.hpp"
#include "tuningSession.hpp"

#include <cmath>
#include <optional>

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

    template<typename Tuple, auto Name>
    static constexpr bool hasTuneableTag()
    {
        return hasTagInTuple<Tuple, Name>();
    }

    template<auto Name>
    static constexpr bool hasFrameTuneable()
    {
        return hasTuneableTag<T_FrameTuneables, Name>();
    }

    template<auto Name>
    static constexpr bool hasUserTuneable()
    {
        return hasTuneableTag<T_UserTuple, Name>();
    }

    template<auto Name>
    static constexpr bool hasTuneable()
    {
        return hasTuneableTag<T_FrameTuneables, Name>() || hasTuneableTag<T_UserTuple, Name>();
    }

    static constexpr bool hasNumBlocksTune()
    {
        return hasFrameTuneable<alpaka::tune::gridSizeName>();
    }

    static constexpr bool hasThreadBlockSizeTune()
    {
        return hasFrameTuneable<alpaka::tune::threadBlockSizeName>();
    }

    static constexpr bool hasNumFramesTune()
    {
        return hasFrameTuneable<alpaka::tune::numFramesName>();
    }

    static constexpr bool hasFrameExtentTune()
    {
        return hasFrameTuneable<alpaka::tune::frameExtentName>();
    }

    constexpr auto& getNumBlocksTune()
    {
        return *getByName<alpaka::tune::gridSizeName>();
    }

    constexpr auto& getThreadBlockSizeTune()
    {
        return *getByName<alpaka::tune::threadBlockSizeName>();
    }

    constexpr auto& getNumFramesTune()
    {
        return *getByName<alpaka::tune::numFramesName>();
    }

    constexpr auto& getFrameExtentTune()
    {
        return *getByName<alpaka::tune::frameExtentName>();
    }

    constexpr auto const& getNumBlocksTune() const
    {
        return *getByName<alpaka::tune::gridSizeName>();
    }

    constexpr auto const& getThreadBlockSizeTune() const
    {
        return *getByName<alpaka::tune::threadBlockSizeName>();
    }

    constexpr auto const& getNumFramesTune() const
    {
        return *getByName<alpaka::tune::numFramesName>();
    }

    constexpr auto const& getFrameExtentTune() const
    {
        return *getByName<alpaka::tune::frameExtentName>();
    }

    template<auto Name>
    constexpr auto& getValue()
    {
        return getByName<Name>()->value;
    }

    template<auto Name>
    constexpr auto const& getValue() const
    {
        return getByName<Name>()->value;
    }

    template<std::size_t I = 0, typename Tuple, auto Name>
    constexpr auto* getByNameImpl(Tuple& tuple)
    {
        if constexpr(I < std::tuple_size_v<Tuple>)
        {
            auto& elem = std::get<I>(tuple);
            if constexpr(elem.tag == Name)
            {
                return &elem;
            }
            else
            {
                return getByNameImpl<I + 1, Tuple, Name>(tuple);
            }
        }
        else
        {
            return &alpaka::tune::noTune; // or nullptr if appropriate
        }
    }

    template<StaticString Tag, typename Tuple, std::size_t I = 0>
    static constexpr auto getNameFromTuple()
    {
        if constexpr(I < std::tuple_size_v<Tuple>)
        {
            using Elem = std::tuple_element_t<I, Tuple>;
            if constexpr(Elem::tag == Tag)
            {
                return Elem::tag;
            }
            else
            {
                return getNameFromTuple<Tag, Tuple, I + 1>();
            }
        }
        else
        {
            return alpaka::tune::empty_name;
        }
    }

    template<StaticString Tag>
    static constexpr auto getName()
    {
        using All = decltype(std::tuple_cat(std::declval<T_FrameTuneables>(), std::declval<T_UserTuple>()));
        return getNameFromTuple<Tag, All>();
    }

    template<std::size_t I = 0, typename Tuple, auto Name>
    constexpr auto* getByNameImpl(Tuple& tuple) const
    {
        if constexpr(I < std::tuple_size_v<Tuple>)
        {
            auto& elem = std::get<I>(tuple);
            if constexpr(elem.tag == Name)
            {
                return &elem;
            }
            else
            {
                return getByNameImpl<I + 1, Tuple, Name>(tuple);
            }
        }
        else
        {
            return &alpaka::tune::noTune; // or nullptr if appropriate
        }
    }

    template<auto Name>
    constexpr auto* getByName()
    {
        auto all = allTuneables();
        return getByNameImpl<0, decltype(all), Name>(all);
    }

    template<auto Name>
    constexpr auto const* getByName() const
    {
        auto all = allTuneables();
        return getByNameImpl<0, decltype(all), Name>(all);
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
