//
// Created by tim on 06.04.25.
//

#ifndef ACTIVEKERNEL_H
#define ACTIVEKERNEL_H
#include "alpaka/tune/active/tuneable.hpp"

#include <alpaka/tune/IO/storageTypes.hpp>
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

template<typename Tuple, typename Func, std::size_t... Indices>
void for_eachTupleDebugImpl(Tuple&& tuple, Func&& func, std::index_sequence<Indices...>)
{
    (func(std::get<Indices>(std::forward<Tuple>(tuple))), ...);
}

template<typename Tuple, typename Func>
void for_eachTupleDebug(Tuple&& tuple, Func&& func)
{
    constexpr auto N = std::tuple_size<std::decay_t<Tuple>>::value;
    for_eachTupleDebugImpl(std::forward<Tuple>(tuple), std::forward<Func>(func), std::make_index_sequence<N>{});
}

template<typename Tuple>
void outTuple(Tuple&& tuple)
{
    for_eachTupleDebug(
        tuple,
        [](auto& tuneable)
        {
            std::cout << "[DEBUG] tuneable Name: " << tuneable.name() << std::endl;
            std::cout << "[DEBUG] Value " << tuneable.value.toString() << std::endl;
            std::cout << "[DEBUG] Start " << tuneable.idxRange.m_begin.toString() << std::endl;
            std::cout << "[DEBUG] End " << tuneable.idxRange.m_end.toString() << std::endl;
            std::cout << "[DEBUG] Stride " << tuneable.idxRange.m_stride.toString() << std::endl;
        });
}

// concretely defined m_run for a tuning session stores extracted
/*
 * this object defines the "core" state of the tuning mechanism, defining type-safe tuning parameters in the form of
 *tuples each subsequent parameter configuration is derived from this active State by working in references of its
 *types ..
 **/
// T_CompileTimeTuple does not actually contain compile time tuneables but runtime tuneables,
// but its used to select a compile time instantiated Kernel based on a internal mapping (T_CompiletimeTuple... ->
// Kernel<Args...>)
template<typename... Ts>
struct FirstOrMonostate;

template<>
struct FirstOrMonostate<>
{
    using type = std::monostate;
};

template<typename T0, typename... Rest>
struct FirstOrMonostate<T0, Rest...>
{
    using type = T0;
};

template<typename... Ts>
using FirstOrMonostate_t = typename FirstOrMonostate<Ts...>::type;

template<typename T_KernelRun>
struct KernelTuningModelView
{
    /// The underlying kernel model type.
    using KernelModel = T_KernelRun;

    /// The shared parameter interfac
    using SharedInterfaceType = typename KernelModel::SharedInterfaceType;

    /**
     * @brief Constructor.
     * @param modelRef Reference to an existing KernelTuningModel instance.
     */
    explicit KernelTuningModelView(KernelModel& modelRef) : model(modelRef)
    {
    }

    /**
     * @brief Returns a reference to all tuneables (user, frame, compile).
     * @return Tuple of tuneable references.
     */
    constexpr auto& allTuneables()
    {
        return model.allTuneables();
    }

    /**
     * @brief Access all parameters based on the specified dimensionTraversePolicy.
     * @return A reference to the shared parameter interface (tuple of TuneableHandles).
     * @throws static_assert if the kernel model does not define a shared interface.
     */
    constexpr SharedInterfaceType& getUniformInterface()
    {
        return model.uniformAccessor();
    }

    /**
     * @brief Extract the current configuration (i.e., tuple of .value from each tuneable).
     * @return A Config object representing the current model state.
     */
    auto toConfig() const
    {
        return model.toConfig();
    }

    /**
     * @brief Apply a configuration from a ConfigEntry wrapper.
     * @tparam T_Config The underlying config tuple type.
     * @param entry The ConfigEntry to apply to the model.
     */
    template<typename T_Config>
    void fromConfig(ConfigEntry<T_Config> const& entry)
    {
        model.fromConfig(entry);
    }

    /**
     * @brief Apply a configuration directly from a Config object.
     * @tparam Ts The types in the config tuple.
     * @param config The Config object to apply.
     */
    template<typename... Ts>
    void fromConfig(Config<Ts...> const& config)
    {
        model.fromConfig(config);
    }

private:
    KernelModel& model;
};

template<
    typename T_UserTuple = std::tuple<>,
    typename T_FrameTuneables = std::tuple<>,
    typename T_CompileTimeTuple = std::tuple<>,
    typename... T_SharedParameterInterface>
struct KernelTuningModel
{
    using T_floating = double_t;
    using T_TuneTuple = decltype(std::tuple_cat(std::declval<T_FrameTuneables>(), std::declval<T_UserTuple>()));
    T_UserTuple m_userTuneables;
    T_FrameTuneables m_frameTuneables;

    T_CompileTimeTuple m_compileTimeTuneables{};
    T_floating metric{};
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    using SharedInterfaceType = FirstOrMonostate_t<T_SharedParameterInterface...>;
    static constexpr bool hasShared = !std::is_same_v<SharedInterfaceType, std::monostate>;

    std::optional<SharedInterfaceType> m_sharedInterface;

    bool resetSignal{false};

    constexpr KernelTuningModel() = default;

    constexpr explicit KernelTuningModel(T_UserTuple userT, T_FrameTuneables frameT)
        : m_userTuneables(std::move(userT))
        , m_frameTuneables(std::move(frameT))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
    {
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, m_userTuneables);
        maxRuns = maxRunsDefault;
    }

    auto& uniformAccessor()
    {
        if constexpr(hasShared)
        {
            if(!m_sharedInterface.has_value())
            {
                std::cout << " it doesnt have value" << std::endl;
            }
            std::cout << " dawok" << std::endl;
            auto& k = m_sharedInterface.value();
            std::cout << " j" << std::endl;
            return k;
        }
        else
        {
            static_assert(hasShared, "No shared interface present in this KernelTuningModel.");
        }
    }

    constexpr auto compileTimeToFlatValueTuple()
    {
        return std::apply(
            [](auto&... t) { return std::tuple_cat(std::make_tuple(t.value)...); },
            m_compileTimeTuneables);
    }

    constexpr explicit KernelTuningModel(T_UserTuple userT, T_FrameTuneables frameT, T_CompileTimeTuple compileT)
        requires(!hasShared)
        : m_userTuneables(std::move(userT))
        , m_frameTuneables(std::move(frameT))
        , m_compileTimeTuneables(std::move(compileT))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())

    {
        std::cout << " for some reason this is called" << std::endl;
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, m_userTuneables);
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, m_compileTimeTuneables);

        maxRuns = maxRunsDefault;
    }

    constexpr explicit KernelTuningModel(T_UserTuple userT, T_FrameTuneables frameT, T_CompileTimeTuple compileT)
        requires hasShared
        : m_userTuneables(std::move(userT))
        , m_frameTuneables(std::move(frameT))
        , m_compileTimeTuneables(std::move(compileT))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
    {
        m_sharedInterface = std::make_optional(makeSharedParameterInterface(*this));
        if(m_sharedInterface.has_value())
        {
            std::cout << " IT HAS VALUE" << std::endl;
        }
        else
        {
            std::cout << " IT HAS NO VALUE" << std::endl;
        }
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, m_userTuneables);
        std::apply([&](auto&... t) { ((maxRunsDefault *= t.numSteps()), ...); }, m_compileTimeTuneables);
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

    // combined tuple using copy by value
    constexpr auto allValues() const
    {
        return std::apply(
            [&](auto const&... frameElems)
            {
                return std::apply(
                    [&](auto const&... userElems)
                    {
                        return std::apply(
                            [&](auto const&... compileElems)
                            {
                                return std::make_tuple(userElems.value..., frameElems.value..., compileElems.value...);
                                // here we actually copy by value
                            },
                            this->m_compileTimeTuneables);
                    },
                    this->m_userTuneables);
            },
            this->m_frameTuneables);
    }

    // access tuple by reference
    constexpr auto allTuneables() &
    {
        return std::apply(
            [this](auto&... frameElems)
            {
                return std::apply(
                    [&](auto&... userElems)
                    {
                        return std::apply(
                            [&](auto&... compileElems)
                            { return std::tie(userElems..., frameElems..., compileElems...); },
                            m_compileTimeTuneables);
                    },
                    m_userTuneables);
            },
            m_frameTuneables);
    }

    auto toConfig() const
    {
        return Config{allValues()};
    }

    constexpr auto allTuneables() const&
    {
        return std::apply(
            [this](auto const&... frameElems)
            {
                return std::apply(
                    [&](auto const&... userElems)
                    {
                        return std::apply(
                            [&](auto const&... compileElems)
                            { return std::tie(userElems..., frameElems..., compileElems...); },
                            m_compileTimeTuneables);
                    },
                    m_userTuneables);
            },
            m_frameTuneables);
    }

    void printFull()
    {
        outTuple(allTuneables());
    };

    template<typename T_Config>
    void fromConfig(ConfigEntry<T_Config> const& config)
    {
        std::apply(
            [&](auto&... tuneables)
            {
                std::apply(
                    [&](auto const&... values) { ((tuneables.value = values), ...); },
                    config.config.getValues());
            },
            allTuneables());
    };

    template<typename... Ts>
    void fromConfig(Config<Ts...> const& config)
    {
        std::apply(
            [&](auto&... tuneables)
            { std::apply([&](auto const&... values) { ((tuneables.value = values), ...); }, config.getValues()); },
            allTuneables());
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

    template<auto... ID>
    constexpr auto getByIDs()
    {
        return std::make_tuple(getByID<ID>()...); // expands each getByID
    }

    template<auto ID>
    constexpr auto* getByID()
    {
        constexpr auto ID_v = static_cast<std::size_t>(ID);
        auto all = allTuneables();
        return getByIDImpl<0, decltype(all), ID_v>(all);
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
    return KernelTuningModel<UserTuple, FilteredTuple>{std::move(userT), std::move(filteredTuple)};
}

//--------------------------------------
// Append tuneable if not present
//--------------------------------------
template<typename ExistingKernel, typename NewTuning>
auto appendTuning(ExistingKernel const& kernel, NewTuning const& newTuning)
{
    using NewTuning_bareT = std::remove_cvref_t<NewTuning>;
    if constexpr(ExistingKernel::template hasFrameTuneable<NewTuning_bareT::tag>())
    {
        std::cout << "TUNER ERROR: tuning already assigned. Skipping.\n";
        return kernel;
    }
    else if constexpr(std::is_same_v<alpaka::tune::NoTune, NewTuning_bareT>)
    {
        return kernel;
    }
    else
    {
        auto newFrameTuple = std::apply(
            [&](auto const&... elems) { return std::make_tuple(elems..., newTuning); },
            kernel.m_frameTuneables);

        return KernelTuningModel{kernel.m_userTuneables, std::move(newFrameTuple)};
    }
}

namespace alpaka::tune
{
    template<typename T_ActiveKernel>
    void recalculateMaxRuns(T_ActiveKernel& active)
    {
        using maxRunsType = decltype(active.maxRuns);
        auto init = maxRunsType{1};
        std::apply([&](auto&... t) { ((init *= t.numSteps()), ...); }, active.m_userTuneables);
        std::apply([&](auto&... t) { ((init *= t.numSteps()), ...); }, active.m_compileTimeTuneables);
        active.maxRuns = init;
        if constexpr(T_ActiveKernel::hasNumBlocksTune())
        {
            recalculateMaxRuns_forTune(active, active.getNumBlocksTune(), "grid");
        }

        if constexpr(T_ActiveKernel::hasThreadBlockSizeTune())
        {
            recalculateMaxRuns_forTune(active, active.getThreadBlockSizeTune(), "block");
        }

        if constexpr(T_ActiveKernel::hasNumFramesTune())
        {
            recalculateMaxRuns_forTune(active, active.getNumFramesTune(), "frame");
        }

        if constexpr(T_ActiveKernel::hasFrameExtentTune())
        {
            recalculateMaxRuns_forTune(active, active.getFrameExtentTune(), "extent");
        }
    }
} // namespace alpaka::tune

#endif // ACTIVEKERNEL_H
