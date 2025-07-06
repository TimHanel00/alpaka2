
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H
#include "../utils/environmentVars.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"
#include "alpaka/tune/utils/Random.h"
#include "alpaka/tune/utils/tupleHelper.h"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/Queue.hpp>
#include <alpaka/tune/active/kernelTuningModel.hpp>
#include <alpaka/tune/traits/traits.hpp>

#include <any>
#include <utility>

template<typename T_Config>
struct EnvironmentState
{
    bool sessionFinished{false};
    uint32_t numberOfCheckedConfigs{0};
    uint32_t numValidConfigs{0};
    uint32_t maxValidEvaluations{0};
    uint32_t maxConfigsTotal{0};
    uint32_t stamp{0};

    bool strategyCriteriaReached()
    {
    }

    ConfigEntry<T_Config> bestConfig;

    bool globalBreakCriteriaFinished(auto const& config)
    {
        return false;
    }

    bool localBreakCriteriaFinished(auto const& config)
    {
        return false;
    }
};

template<typename TuneablesTuple, typename ExpandedTuple>
void printTuneableDimensions(TuneablesTuple const& allTuneables, ExpandedTuple const& expandedTuneables)
{
    for_each_enumerate(
        expandedTuneables,
        [&](auto const& wrapper, std::size_t)
        {
            visitIndex(
                wrapper.id,
                allTuneables,
                [&](auto const& tuneable)
                {
                    std::cout << tuneable.name() << "_" << wrapper.dim << " = [ ";

                    for(std::size_t i = 0; i < wrapper.list.size(); ++i)
                    {
                        std::cout << wrapper.list[i];
                        if(i + 1 < wrapper.list.size())
                            std::cout << ", ";
                    }

                    std::cout << " ]" << std::endl;
                });
        });
}

template<typename... Tuneables>
void makeListsForAllTuneables(std::tuple<Tuneables...>&& allTuneables)
{
    std::apply(
        [](auto&... tuns)
        {
            (void) std::initializer_list<int>{
                (tuns.valueList = tuns.makeList(), 0)... // discard result
            };
        },
        allTuneables);
}

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
    std::cout << initialMaxRuns << " runs initial\n";

    auto expandedTuneables = expand(allTuneables);

    while(initialMaxRuns > alpaka::tune::getMaxConfigs())
    {
        std::vector<std::pair<std::size_t, std::size_t>> maxRunsVec;

        for_each_enumerate(
            expandedTuneables,
            [&](auto& wrapper, std::size_t index) { maxRunsVec.emplace_back(wrapper.list.size(), index); });

        std::sort(maxRunsVec.begin(), maxRunsVec.end(), std::greater<>());

        std::size_t maxIndex = maxRunsVec.front().second;

        bool shouldContinue = true;

        visitIndex(
            maxIndex,
            expandedTuneables,
            [&](auto& wrapper)
            {
                if(wrapper.list.size() < 2)
                {
                    shouldContinue = false;
                    return;
                }

                bool removed = removeAllMatchingIndices(
                    allTuneables,
                    maxIndex,
                    expandedTuneables,
                    [](std::size_t i) { return (i & 1) == 1; } // odd indices
                );

                // fallback if nothing removed
                if(!removed)
                {
                    removeAllMatchingIndices(
                        allTuneables,
                        maxIndex,
                        expandedTuneables,
                        [](std::size_t i) { return (i & 1) == 0; } // even indices
                    );
                }
            });

        if(!shouldContinue)
            break;

        initialMaxRuns = 1;
        for_each(allTuneables, [&](auto& tunable) { initialMaxRuns *= tunable.numSteps(); });
    }
    printTuneableDimensions(allTuneables, expandedTuneables);
}
template<typename T>
struct KernelModelForSPI;
#define REGISTER_SPI_TYPE(S_Type, ModelType)                                                                          \
    template<>                                                                                                        \
    struct KernelModelForSPI<S_Type>                                                                                  \
    {                                                                                                                 \
        using type = ModelType;                                                                                       \
    };

namespace alpaka::tune
{
    // #define DEBUG_Singleton
    template<
        typename T_Config,
        typename T_ConfigDescriptor,
        typename T_NumFrames,
        typename T_FrameExtent,
        typename T_Strategy,
        typename T_MetricInterface,
        typename T_Constraints,
        typename T_KernelTuningModel>
    class tuningEnvironment
    {
    public:
        using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>;

        T_Strategy env_strategy;
        T_MetricInterface env_metricInterface;
        T_Constraints env_constraints;
        T_KernelTuningModel env_kernelTuningPtr;
        KernelData<T_Config, T_ConfigDescriptor> env_kernelData;
        EnvironmentState<T_Config> environmentState;
        alpaka::tune::ConfigQueue<ConfigEntry<T_Config>> env_config_queue;
        tuningEnvironment(tuningEnvironment const&) = delete;
        tuningEnvironment& operator=(tuningEnvironment const&) = delete;
        tuningEnvironment(tuningEnvironment&&) = delete;
        tuningEnvironment& operator=(tuningEnvironment&&) = delete;

        auto& getConfigStorage()
        {
            return env_kernelData.configEntries;
        }

        tuningEnvironment(
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

        {
            auto& history = alpaka::tune::TuningHistory::get(filename);
            history.loadConfig(env_kernelData);
            // 1. Clamp to spec
            alpaka::tune::clampToSpec(device_, frameSpec_, *env_kernelTuningPtr);
            // 2. Generate value lists
            makeListsForAllTuneables(env_kernelTuningPtr->allTuneables());

            // 3. Initial max runs
            alpaka::tune::recalculateMaxRuns(*env_kernelTuningPtr);
            shrinkTuningSpace(env_kernelTuningPtr->allTuneables(), env_kernelTuningPtr->maxRuns);
            alpaka::tune::recalculateMaxRuns(*env_kernelTuningPtr);
            environmentState.bestConfig = getConfigStorage().getOrCreate(env_kernelTuningPtr->toConfig());
            environmentState.maxConfigsTotal = env_kernelTuningPtr->maxRuns;
            environmentState.maxValidEvaluations = alpaka::tune::getMaxRuns();
        }

        // Prevent copy/move
    };

    template<typename T_Env>
    class TuningContextManager : public T_Env
    {
    public:
        using Base = T_Env;
        using Base::Base; // inherit constructor

        template<typename... T_Args>
        void launch(T_Args&&... launchArgs)
        {
            if(this->environmentState.checkGlobalBreakCriteria())
            {
                executeBestConfig(std::forward<T_Args...>(launchArgs...));
                return;
            }

            if(handleFullQueue(std::forward<T_Args...>(launchArgs...)))
                return;

            while(!this->environmentState.strategyCriteriaReached())
            {
                if(this->env_config_queue.empty())
                {
                    executeBestConfig();
                    return;
                }

                auto oldConfig = this->env_kernelTuningPtr->toConfig();
                auto newConfig = oldConfig;

                while(this->getConfigStorage().contains(newConfig) || violatesConstraint(newConfig))
                {
                    this->env_strategy(
                        this->env_metricInterface,
                        KernelTuningModelView(*this->env_kernelTuningPtr),
                        this->getConfigStorage(),
                        this->environmentState);

                    newConfig = this->env_kernelTuningPtr->toConfig();
                    if(this->environmentState.strategyCriteriaReached())
                    {
                        if(emptyTheQueue(std::forward<T_Args...>(launchArgs...)))
                            return;
                    }
                }

                this->env_config_queue.push(newConfig);

                if(handleFullQueue(std::forward<T_Args...>(launchArgs...)))
                    return;
            }
        }

    private:
        template<typename... T_Args>
        bool emptyTheQueue(T_Args&&... launchArgs)
        {
            while(!this->env_config_queue.empty())
            {
                auto& config = this->env_config_queue.get();
                if(this->environmentState.checkLocalBreakCriteria(config))
                {
                    this->env_config_queue.pop();
                }
                else
                {
                    applyAndExecute(std::forward<T_Args...>(launchArgs...), config);
                    update(config);
                    return true;
                }
            }
            return false;
        }

        template<typename... T_Args>
        bool handleFullQueue(T_Args&&... launchArgs)
        {
            while(this->env_config_queue.full())
            {
                auto& config = this->env_config_queue.get();
                if(this->environmentState.checkLocalBreakCriteria(config))
                {
                    this->env_config_queue.pop();
                }
                else
                {
                    applyAndExecute(std::forward<T_Args...>(launchArgs...), config);
                    update(config);
                    return true;
                }
            }
            return false;
        }

        template<typename... T_Args>
        void executeBestConfig(T_Args&&... launchArgs)
        {
            auto& bestConfig = this->environmentState.bestConfig;
            applyAndExecute(std::forward<T_Args...>(launchArgs...), bestConfig);
            update(bestConfig);
        }

        bool violatesConstraint(auto const& config)
        {
            auto& stored = this->getConfigStorage().getOrCreate(config);
            if(stored.state == ConfigState::Dummy)
            {
                return true;
            }

            bool valid = true;
            int index = 0;
            for_each(
                this->env_constraints,
                [&index, &valid, &run = *this->env_kernelTuningPtr](auto& constraint)
                {
                    auto constraintValid = constraint.template operator()<decltype(run)>(run);
                    valid = valid && constraintValid;
                });

            if(!valid)
            {
                std::cout << " constraint violated for : " << stored.config.toString() << std::endl;
                using T_state = decltype(stored.state);
                ++this->environmentState.numberOfCheckedConfigs;
                stored.getMetrics().clear();
                stored.stamp = -1;
                stored.state = T_state::Dummy;
                stored.fullFlag = true;
                stored.nr_runs = std::numeric_limits<decltype(stored.nr_runs)>::max();
                return true;
            }

            return false;
        }

        template<typename T_Queue, typename T_Exec, typename T_Spec, typename T_Kernelbundle>
        void applyAndExecute(
            T_Queue&& queue,
            T_Exec&& exec,
            T_Spec&& spec,
            T_Kernelbundle&& kernelbundle,
            auto const& config)
        {
            auto& run = *this->env_kernelTuningPtr;
            run.fromConfig(config);
            applyCustomThreadSpec(run, spec);
            auto bundle = recreate(kernelbundle, run.m_userTuneables);

            trait::callPreProcessing(run, spec, this->env_metricInterface, bundle);

            using KernelFn = typename decltype(this->env_kernelTuningPtr->kernelBundle)::KernelFn;
            std::cout << " try to launch kernel with " << run.toConfig().toString() << std::endl;

            if constexpr(!trait::hasUserDefinedCTuneable<KernelFn>::value)
            {
                this->env_metricInterface.start(run, spec);
                queue.enqueue(exec, spec, bundle);
                onHost::wait(queue);
                this->env_metricInterface.end(run, spec);
            }
            else
            {
                std::size_t i = trait::getRtimeIndexMap(bundle)[run.compileTimeToFlatValueTuple()];
                static auto variants =
                    typename trait::RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants{};

                alpaka::tune::runtime_Kernel_dispatch(
                    i,
                    variants,
                    [&](auto&& element)
                    {
                        auto newBundle = std::apply(
                            [&element]<typename... T0>(T0&&... args)
                            { return KernelBundle{element, std::forward<T0>(args)...}; },
                            bundle.m_args);

                        this->env_metricInterface.start(run, spec);
                        queue.enqueue(exec, spec, newBundle);
                        onHost::wait(queue);
                        this->env_metricInterface.end(run, spec);
                    });
            }
            trait::callPostProcessing(run, spec, this->env_metricInterface, bundle);
        }

        void update(auto const& config)
        {
            auto& run = *this->env_kernelTuningPtr;
            std::cout << " ran config: " << config.toHash() << " time " << run.metric << std::endl;
            auto& stored = this->getConfigStorage().getOrCreate(config);

            switch(stored.state)
            {
            case ConfigState::Uninitialized:
                stored.stamp = this->env_kernelData.highestStamp + this->environmentState.stamp++;
                break;
            case ConfigState::Dummy:
                return;
            default:
                std::cout << " config has state: " << config.toString()
                          << " state: " << static_cast<std::size_t>(stored.state) << std::endl;
                break;
            }

            bool flagPre = stored.fullFlag;
            stored.pushMetric(run.metric);
            bool flagPost = stored.fullFlag;

            if(!alpaka::tune::hasRunsPerConfig_Env())
            {
                if(flagPre != flagPost)
                {
                    ++this->environmentState.numberOfCheckedConfigs;
                    ++this->environmentState.numValidConfigs;
                    assignBestIfBetter<typename Base::env_metricInterface>(this->environmentState.bestConfig, stored);
                }
            }
            else
            {
                if(flagPre != flagPost)
                {
                    stored.fullFlag = false;
                }
                if(stored.nr_runs >= alpaka::tune::getRunsPerConfig())
                {
                    if(!stored.fullFlag)
                    {
                        ++this->environmentState.numberOfCheckedConfigs;
                        ++this->environmentState.numValidConfigs;
                    }
                    assignBestIfBetter<typename Base::env_metricInterface>(this->environmentState.bestConfig, stored);
                    stored.fullFlag = true;
                }
            }
        }
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
    std::cout << tuneable.value.toString() << std::endl;
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
                std::cout << " has range ENV" << tuneable.hasRange << std::endl;
                std::cout << " has range IN" << tuneable.hasRange << " " << value.toString() << std::endl;
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
auto makeConformToFrameSpec(T_frameSpec& spec, KernelTuningModel<T_Args...>& kernelRun)
{
    // KernelTuningModel m_run;
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
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier>
auto createTuningEnvironment(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    T_Strategy& strategy,
    T_MetricInterface& metric_interface,
    T_Constraints& constraint,
    T_Run& run,
    T_SessionSpecifier& sessionSpecifier,
    std::string const& filename)
{
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
    auto CTuneableBundle = alpaka::tune::trait::constructRuntimeCtuneablesForActivKernel(bundle);
    auto userTuple = extractTuneables(bundle);

    // Combine into kernel model
    auto completeRun = KernelTuningModel{userTuple, newRun.m_frameTuneables, CTuneableBundle};

    using T_Config = decltype(completeRun.toConfig());
    auto env_kernelData = createKernelDataFromModel(
        completeRun,
        alpaka::core::demangledName(device),
        alpaka::core::demangledName(exec),
        alpaka::core::demangledName<decltype(bundle)>(),
        sessionSpecifier);
    using kernelModel = decltype(completeRun);


    using T_sharedParmeterInterface = decltype(makeSharedParameterInterface(completeRun));
    using model = KernelTuningModel<
        decltype(userTuple),
        decltype(newRun.m_frameTuneables),
        decltype(CTuneableBundle),
        T_sharedParmeterInterface>;
    auto activePtr = std::make_unique<model>(userTuple, newRun.m_frameTuneables, CTuneableBundle);
    // Final types deduced for environment
    using tuningEnvironmentType = alpaka::tune::tuningEnvironment<
        T_Config,
        decltype(env_kernelData.descriptor),
        T_NumFrames,
        T_FrameExtent,
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
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier>
auto& getTuningEnvironment(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
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
