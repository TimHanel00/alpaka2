
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H

#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/core/peripherals/queue.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>
#include <alpaka/tune/traits/traits.hpp>
#include <alpaka/tune/tuneable/kernelTuningModel.hpp>
#include <alpaka/tune/utils/Random.hpp>
#include <alpaka/tune/utils/tupleHelper.hpp>

#include <utility>
#define Tuner_MaxConsecutiveStrategyFailures 20000

template<typename... Tuneables>
auto expand(std::tuple<Tuneables...>& tuneables)
{
    return [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        return std::tuple_cat(std::get<Is>(tuneables).expand(Is)...);
    }(std::make_index_sequence<sizeof...(Tuneables)>{});
}

template<typename TuneablesTuple, typename WrapperTuple, typename Predicate>
bool removeAllMatchingIndices(
    TuneablesTuple& allTuneables,
    std::size_t wrapperIndex,
    WrapperTuple& wrappers,
    Predicate shouldRemoveIndex)
{
    bool removedAny = false;

    visitIndex(
        wrapperIndex,
        wrappers,
        [&](auto& wrapper)
        {
            visitIndex(
                wrapper.id,
                allTuneables,
                [&](auto& tuneable)
                {
                    for(std::size_t i = wrapper.list.size(); i-- > 0;)
                    {
                        if(shouldRemoveIndex(i))
                        {
                            if(tuneable.removeIfValid(i, wrapper.dim))
                            {
                                removedAny = true;
                            }
                        }
                    }
                });
        });

    return removedAny;
}

template<typename... Tuneables>
void shrinkTuningSpace(std::tuple<Tuneables...>&& allTuneables, std::size_t initialMaxRuns)
{
    auto expandedTuneables = expand(allTuneables);

    while(initialMaxRuns > alpaka::tune::getMaxConfigs())
    {
        //@TODO reimplement with new tuneable interface, ensure m_numValues is recalculated, use model as parameter
        // std::vector<std::pair<std::size_t, std::size_t>> maxRunsVec;
        //
        // for_each_enumerate(
        //     expandedTuneables,
        //     [&](auto& wrapper, std::size_t index) { maxRunsVec.emplace_back(wrapper.list.size(), index); });
        //
        // std::sort(maxRunsVec.begin(), maxRunsVec.end(), std::greater<>());
        //
        // std::size_t maxIndex = maxRunsVec.front().second;
        //
        // bool shouldContinue = true;
        //
        // visitIndex(
        //     maxIndex,
        //     expandedTuneables,
        //     [&](auto& wrapper)
        //     {
        //         if(wrapper.list.size() < 2)
        //         {
        //             shouldContinue = false;
        //             return;
        //         }
        //
        //         bool removed = removeAllMatchingIndices(
        //             allTuneables,
        //             maxIndex,
        //             expandedTuneables,
        //             [](std::size_t i) { return (i & 1) == 1; } // odd indices
        //         );
        //
        //         // fallback if nothing removed
        //         if(!removed)
        //         {
        //             removeAllMatchingIndices(
        //                 allTuneables,
        //                 maxIndex,
        //                 expandedTuneables,
        //                 [](std::size_t i) { return (i & 1) == 0; } // even indices
        //             );
        //         }
        //     });
        //
        // if(!shouldContinue)
        //     break;
        //
        // initialMaxRuns = 1;
        // for_each(allTuneables, [&](auto& tunable) { initialMaxRuns *= tunable.numSteps(); });
    }
#ifdef debug
    printTuneableDimensions(allTuneables, expandedTuneables);
#endif
}

namespace alpaka::tune
{
    // #define DEBUG_Singleton
    template<
        typename T_Config,
        typename T_ConfigDescriptor,
        typename T_NumFrames,
        typename T_FrameExtent,
        typename T_ThreadExtent,
        typename T_Strategy,
        typename T_MetricInterface,
        typename T_Constraints,
        typename T_KernelTuningModel>
    class TuningContext
    {
    public:
        using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent, T_ThreadExtent>;
        using T_MetricInterfaceType = T_MetricInterface;
        T_Strategy env_strategy;
        T_MetricInterface env_metricInterface;
        T_Constraints env_constraints;
        T_KernelTuningModel env_kernelTuningPtr;
        KernelData<T_Config, T_ConfigDescriptor> env_kernelData;
        EnvironmentState<T_Config> environmentState;
        ConfigQueue<ConfigEntry<T_Config>> env_config_queue;
        TuningContext(TuningContext const&) = delete;
        TuningContext& operator=(TuningContext const&) = delete;
        TuningContext(TuningContext&&) = delete;
        TuningContext& operator=(TuningContext&&) = delete;
        TuningHistory& env_history;

        auto& getConfigStorage()
        {
            return env_kernelData.configEntries;
        }

        bool violatesConstraint(auto const& config)
        {
            auto& stored = this->getConfigStorage().getOrCreate(config);
            if(stored.state == ConfigState::Dummy)
            {
#ifdef Debug
                std::cout << "[violatesConstraint] Already dummy: " << stored.Config.toString() << "\n";
#endif
                return true;
            }
            using T_KernelTuningModel_ = decltype(*this->env_kernelTuningPtr);
            bool valid = true;
            for_each(
                this->env_constraints,
                [&](auto& constraint)
                {
                    if(!constraint.template operator()<T_KernelTuningModel_>(*this->env_kernelTuningPtr))
                        valid = false;
                });

            if(!valid)
            {
                stored.getMetrics().clear();
                stored.stamp = -1;
                stored.state = ConfigState::Dummy;
                stored.fullFlag = true;
                stored.nr_runs = std::numeric_limits<decltype(stored.nr_runs)>::max();
#ifdef Debug
                std::cout << "[violatesConstraint] Marked invalid: " << stored.Config.toString() << "\n";
#endif
                ++this->environmentState.numberOfCheckedConfigs;
                return true;
            }

            return false;
        }

        TuningContext(
            KernelData<T_Config, T_ConfigDescriptor>&& env_kernelData_,
            auto& device_,
            FrameSpecType&& frameSpec_,
            T_Strategy&& strategy_,
            T_MetricInterface&& metric_interface_,
            T_Constraints&& constraints_,
            T_KernelTuningModel&& activeRun_,

            std::string const& filename)
            : env_kernelData(std::forward<KernelData<T_Config, T_ConfigDescriptor>>(env_kernelData_))
            , env_strategy(std::forward<T_Strategy>(strategy_))
            , env_metricInterface(std::forward<T_MetricInterface>(metric_interface_))
            , env_constraints(std::forward<T_Constraints>(constraints_))
            , env_kernelTuningPtr(std::forward<T_KernelTuningModel>(activeRun_))
            , env_history(TuningHistory::get(filename))

        {
            env_history.loadConfig<T_MetricInterface>(env_kernelData, environmentState);
            alpaka::tune::recalculateMaxRuns(*env_kernelTuningPtr);
            getRunsPerConfig();
            getMaxRuns();
            getMaxConfigs(); // make sure all environment variables are called atleast once (for has.. to work)


            // 1. Clamp to spec
            // 2. Generate value lists
            auto initConfigTmp = env_kernelTuningPtr->toConfig();

            // 3. Initial max runs
            if(!getConfigStorage().contains(initConfigTmp))
            {
                environmentState.bestConfig = std::ref(getConfigStorage().getOrCreate(initConfigTmp));
                ++environmentState.numberOfCheckedConfigs;
                if(!violatesConstraint(initConfigTmp))
                {
                    environmentState.getBestConfig().stamp
                        = this->env_kernelData.highestStamp + this->environmentState.stamp++;

                    ++environmentState.numValidConfigs;
                }
            }
            if(!environmentState.bestConfig.has_value())
                environmentState.bestConfig = std::ref(getConfigStorage().getOrCreate(initConfigTmp));


            env_config_queue.push_back(
                environmentState.getBestConfig()); // if its full or invalid it will simply get dropped
            environmentState.maxConfigsTotal = std::min(getMaxCheckConfigs(), env_kernelTuningPtr->maxRuns);
            environmentState.maxValidEvaluations = std::min(getMaxCheckConfigs(), getMaxRuns());
        }

        // Prevent copy/move
    };
} // namespace alpaka::tune

template<typename T_Vec>
auto makeConformToTVec(T_Vec const&, alpaka::tune::NoTune const&)

{
    return alpaka::tune::NoTune{};
}

/*
 * ensures that a a user defined tuning conforms to the framespec types and I know its ugly
 *
 */
template<typename T_Vec, typename T_Tuneable>
auto makeConformToTVec(T_Vec const& vec, T_Tuneable& tuneable)
{
    /*
    using Valuetype = T;
    using dimensionTraversePolicy_type = dimensionTraversePolicy;
    static constexpr std::size_t tag = getId<ID>();
    */
#ifdef Debug
    std::cout << tuneable.value.toString() << std::endl;
#endif
    using T_TuneableVec = typename T_Tuneable::ValueType;
    using T_traversePolicy = typename T_Tuneable::dimensionTraversePolicy_type;
    constexpr auto tuneable_ID = T_Tuneable::tag;
    constexpr std::size_t targetDim = T_Vec::dim();
    constexpr std::size_t sourceDim = ALPAKA_TYPEOF(tuneable.value)::dim();

    if constexpr(sourceDim != targetDim)
    {
        if(!tuneable.userDef)
        {
            T_Vec ones = T_Vec::all(1);
            auto ret
                = alpaka::tune::Tuneable<T_Vec, tuneable_ID, T_traversePolicy>(alpaka::IdxRange{ones, vec, ones}, vec);
            ret.userDef = false;
            ret.hasRange = tuneable.hasRange;
            return ret;
        }
        std::string s = std::string(tuneable.name());
        throw std::runtime_error("The dimension of " + s + " must match the dimension of the threadSpec.");
    }
    else
    {
        if(tuneable.userDef)
        {
            // Fallback case when targetDim == sourceDim
            T_Vec value, begin, end, stride;
            if constexpr(std::is_convertible_v<T_Vec, T_TuneableVec>)
            {
                auto retTuneable = alpaka::tune::Tuneable<T_Vec, tuneable_ID, T_traversePolicy>(
                    tuneable.idxRange,
                    tuneable.value,
                    tuneable.name());

                retTuneable.inputList.resize(tuneable.inputList.size());
                std::transform(
                    tuneable.inputList.begin(),
                    tuneable.inputList.end(),
                    retTuneable.inputList.begin(),
                    [](auto const& x) { return static_cast<T_Vec>(x); });
#ifdef Debug
                std::cout << " has range ENV" << tuneable.hasRange << std::endl;
                std::cout << " has range IN" << tuneable.hasRange << " " << value.toString() << std::endl;
#endif
                retTuneable.hasRange = tuneable.hasRange;
                return retTuneable;
            }
            else
            {
                throw std::runtime_error("Type of frameSpec is not convertible to corresponding tuneables.");
            }
        }

        T_Vec ones = T_Vec::all(1);
        auto ret = alpaka::tune::Tuneable<ALPAKA_TYPEOF(vec), tuneable_ID, T_traversePolicy>(
            alpaka::IdxRange{ones, vec, ones},
            vec);
        ret.userDef = false;
        ret.hasRange = tuneable.hasRange;
        return ret;
    }
}

template<typename T_frameSpec, typename... T_Args>
auto makeConformToFrameSpec(T_frameSpec& spec, ConfigDescriptor<T_Args...>& kernelRun)
{
    // ConfigDescriptor m_run;
    // auto h = makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune());
    return makeActiveKernel(
        kernelRun.m_userTuneables,
        makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune()),

        makeConformToTVec(spec.m_frameExtent, kernelRun.getFrameExtentTune()),
        makeConformToTVec(spec.m_threadSpec.m_numBlocks, kernelRun.getNumBlocksTune()),
        makeConformToTVec(spec.m_threadSpec.m_numThreads, kernelRun.getThreadBlockSizeTune()));
}

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_ThreadSpec,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier>
auto createTuningEnvironment(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent, T_ThreadSpec> const& spec,
    T_KernelBundle bundle,
    T_Strategy& strategy,
    T_MetricInterface& metric_interface,
    T_Constraints& constraint,
    T_Run& run,
    T_SessionSpecifier& sessionSpecifier,
    std::string const& filename)
{
    alpaka::tune::benchmark::phaseAccessor(0);
    // Apply spec-based conforming
    auto activeRun = makeConformToFrameSpec(spec, run);

    // Apply HW-specific constraints
    auto retPair = alpaka::tune::applyHwConstraints(device, exec, spec, activeRun);
    auto newFrameSpec = retPair.first;
    auto newRun = retPair.second;

#ifdef DEBUG_Singleton
    newRun.getNumBlocksTune().idxRange.print();
#endif

    // Extract compile-time tuneables for bundle
    auto CTuneableBundle = alpaka::tune::trait::constructRuntimeCtuneablesForActiveKernel(bundle);
    auto userTuple = extractTuneables(bundle);
    // Combine into kernel model
    auto completeRun = ConfigDescriptor{userTuple, newRun.m_frameTuneables, CTuneableBundle};

    using T_Config = decltype(completeRun.toConfig());

    using kernelModel = decltype(completeRun);
    //---- reconfigure kerneltuningModel --- //
    alpaka::tune::clampToSpec(device, newFrameSpec, completeRun);
#ifdef Debug
    std::cout << " bef make Lists " << std::endl;
#endif
    makeListsForAllTuneables(completeRun.allTuneables()); // make lists for all tuneables
#ifdef Debug
    std::cout << " bef first calcRuns " << std::endl;
#endif
    alpaka::tune::recalculateMaxRuns(completeRun);
#ifdef Debug
    std::cout << " bef shrink " << std::endl;
#endif
    shrinkTuningSpace(completeRun.allTuneables(), completeRun.maxRuns);
#ifdef Debug
    std::cout << " aft shrink " << std::endl;
    std::cout << " aft calcMaxRuns " << std::endl;
#endif
    alpaka::tune::recalculateMaxRuns(completeRun);

    //---- reconfigure kerneltuningModel --- //
    using T_sharedParmeterInterface = decltype(makeSharedParameterInterface(completeRun));
    using model = ConfigDescriptor<
        decltype(completeRun.m_userTuneables),
        decltype(completeRun.m_frameTuneables),
        decltype(completeRun.m_compileTimeTuneables),
        T_sharedParmeterInterface>;
    auto activePtr = std::make_unique<model>(
        completeRun.m_userTuneables,
        completeRun.m_frameTuneables,
        completeRun.m_compileTimeTuneables,
        SharedTag{});
    // Final types deduced for environment
    auto env_kernelData = createKernelDataFromModel(
        *activePtr,
        alpaka::onHost::demangledName(device),
        alpaka::onHost::demangledName(exec),
        alpaka::onHost::demangledName<T_KernelBundle>(),
        sessionSpecifier);
    using tuningEnvironmentType = alpaka::tune::TuningContext<
        T_Config,
        decltype(env_kernelData.descriptor),
        T_NumFrames,
        T_FrameExtent,
        T_ThreadSpec,
        T_Strategy,
        T_MetricInterface,
        T_Constraints,
        decltype(activePtr)>;
    using T_Context = alpaka::tune::TuningContextManager<tuningEnvironmentType>;
    return std::make_unique<T_Context>(
        std::move(env_kernelData),
        device,
        std::move(newFrameSpec),
        std::move(strategy),
        std::move(metric_interface),
        std::move(constraint),
        std::move(activePtr),
        filename);
}

// Static wrapper version
inline std::string flattenSessionSpecifier(std::vector<std::string> const& vec)
{
    std::string result;
    for(auto const& s : vec)
    {
        result += s;
    }
    return result;
}

template<
    typename T_Queue,
    typename T_Exec,
    typename T_TuneSpec,
    typename T_FrameExtent,
    typename T_ThreadSpec,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier>
auto& getTuningEnvironment(
    T_Queue queue,
    T_Exec exec,
    T_TuneSpec const& spec,
    T_KernelBundle bundle,
    T_Strategy& strategy,
    T_MetricInterface& metric_interface,
    T_Constraints& constraint,
    T_Run& run,
    T_SessionSpecifier& sessionSpecifier,
    std::string const& filename)
{
    using EnvPtr = decltype(createTuningEnvironment(
        device,
        exec,
        spec,
        bundle,
        strategy,
        metric_interface,
        constraint,
        run,
        sessionSpecifier,
        filename));

    static std::unordered_map<std::string, EnvPtr> singletonMap;

    std::string const key = flattenSessionSpecifier(sessionSpecifier);
    auto [it, inserted] = singletonMap.try_emplace(
        key,
        createTuningEnvironment(
            device,
            exec,
            spec,
            bundle,
            strategy,
            metric_interface,
            constraint,
            run,
            sessionSpecifier,
            filename));

    return it->second;
}

#endif // KERNELSINGLETON_H
