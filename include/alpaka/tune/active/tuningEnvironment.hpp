
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H

#include "../utils/environmentVars.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"
#include "alpaka/tune/utils/Random.h"
#include "alpaka/tune/utils/tupleHelper.h"
#include "tuningSession.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/Queue.hpp>
#include <alpaka/tune/active/kernelTuningModel.hpp>
#include <alpaka/tune/traits/traits.hpp>
#include <alpaka/tune/utils/compileTimeTemplates.hpp>

#include <any>
#include <utility>
#define Tuner_MaxConsecutiveStrategyFailures 20000

template<typename T_Config>
struct EnvironmentState
{
    bool sessionFinished{false};
    bool strategyFinished{false};
    uint32_t numberOfCheckedConfigs{0};
    uint32_t numValidConfigs{0};
    uint32_t maxValidEvaluations{UINT32_MAX};
    uint32_t maxConfigsTotal{0};
    uint32_t stamp{0};
    uint32_t strategyLimit = Tuner_MaxConsecutiveStrategyFailures;

    auto setStrategyFinished() -> void
    {
        strategyFinished = true;
    }

    bool strategyCriteriaReached(std::optional<uint32_t> currentIndex = std::nullopt)
    {
        if(currentIndex.has_value())
        {
            if(currentIndex >= strategyLimit)
            {
                strategyFinished = true;
            }
        }
        return strategyFinished;
    }

    std::optional<std::reference_wrapper<ConfigEntry<T_Config>>> bestConfig;

    auto& getBestConfig()
    {
        return bestConfig.value().get();
    }

    uint32_t getMaxEvals() const
    {
        return std::min(maxValidEvaluations, maxConfigsTotal);
    }

    bool globalBreakCriteriaFinished()
    {
        return numValidConfigs >= maxValidEvaluations || numberOfCheckedConfigs >= maxConfigsTotal;
    }

    bool localBreakCriteriaFinished(auto const& config)
    {
        return config.fullFlag;
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
        typename T_Strategy,
        typename T_MetricInterface,
        typename T_Constraints,
        typename T_KernelTuningModel>
    class tuningEnvironment
    {
    public:
        using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>;
        using T_MetricInterfaceType = T_MetricInterface;
        T_Strategy env_strategy;
        T_MetricInterface env_metricInterface;
        T_Constraints env_constraints;
        T_KernelTuningModel env_kernelTuningPtr;
        KernelData<T_Config, T_ConfigDescriptor> env_kernelData;
        EnvironmentState<T_Config> environmentState;
        ConfigQueue<ConfigEntry<T_Config>> env_config_queue;
        tuningEnvironment(tuningEnvironment const&) = delete;
        tuningEnvironment& operator=(tuningEnvironment const&) = delete;
        tuningEnvironment(tuningEnvironment&&) = delete;
        tuningEnvironment& operator=(tuningEnvironment&&) = delete;
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
                std::cout << "[violatesConstraint] Already dummy: " << stored.config.toString() << "\n";
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
                std::cout << "[violatesConstraint] Marked invalid: " << stored.config.toString() << "\n";
#endif
                ++this->environmentState.numberOfCheckedConfigs;
                return true;
            }

            return false;
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

#define BestMeasurements 10

    template<typename EnvBase>
    class TuningContextManager : public EnvBase
    {
    public:
        using Base = EnvBase;
        using Base::Base; // inherit constructor
        int bestCounter = 0;
        bool readyForTerminate = false;

        template<typename... T_Args>
        void launch(T_Args&&... launchArgs)
        {
#ifdef Debug
            std::cout << "[launch] Entered launch function.\n";

#endif
            std::cout << "[Num evaluations]" << "," << this->environmentState.numValidConfigs << "\n";
            // std::cout << " kerneltuning config " << this->env_kernelTuningPtr->toConfig().toString() << std::endl;
            if(this->environmentState.sessionFinished)
            {
#ifdef Debug
                std::cout << "[launch] Session has finished. bestCoutner:" << bestCounter << "\n";
#endif
                if(bestCounter == BestMeasurements)
                {
#ifdef Debug
                    std::cout << "[launch] BestMeasurements reached. Marking readyForTerminate.\n";
#endif
                    readyForTerminate = true;
                    // this->env_history.storeConfig(this->env_kernelData);
                    tune::benchmark::phaseAccessor(4);
                    executeBestConfig(std::forward<T_Args>(launchArgs)...);
                    bestCounter++;
                    return;
                }
#ifdef Debug
                std::cout << "[launch] Executing best config again (count " << bestCounter << ").\n";
#endif
                executeBestConfig(std::forward<T_Args>(launchArgs)...);
                tune::benchmark::phaseAccessor(3);
                bestCounter++;
                return;
            }

            benchmark::phaseAccessor(2);

            if(this->environmentState.globalBreakCriteriaFinished())
            {
#ifdef Debug
                std::cout << "[launch] Global break criteria reached. Emptying queue.\n";
#endif
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }

            if(handleFullQueue(std::forward<T_Args>(launchArgs)...))
            {
#ifdef Debug
                std::cout << "[launch] Full queue handled.\n";
#endif
                return;
            }

#ifdef Debug
            std::cout << "[launch] Fetching current config from kernel tuner.\n";
#endif
            auto currentConfig = this->env_kernelTuningPtr->toConfig();
            int i = 0;

            while(!this->environmentState.strategyCriteriaReached(i++)
                  && !this->environmentState.globalBreakCriteriaFinished()
                  && (this->getConfigStorage().contains(currentConfig) || Base::violatesConstraint(currentConfig)))
            {
#ifdef Debug
                std::cout
                    << "[launch] Strategy criteria not reached or config invalid/redundant. Creating new config.\n";
#endif
                benchmark::phaseAccessor(5);
                auto view = KernelTuningModelView(*this->env_kernelTuningPtr);
                this->env_strategy(this->env_metricInterface, view, this->getConfigStorage(), this->environmentState);
                currentConfig = this->env_kernelTuningPtr->toConfig();
            }

            if(this->environmentState.strategyCriteriaReached()
               || this->environmentState.globalBreakCriteriaFinished())
            {
#ifdef Debug
                std::cout << "[launch] Strategy criteria or global break reached after loop. Emptying queue.\n";
#endif
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }

#ifdef Debug
            std::cout << "[launch] Pushing new config: " << currentConfig.toString() << "\n";
#endif
            auto& newEntry = this->getConfigStorage().getOrCreate(currentConfig);
            this->env_config_queue.push_back(newEntry);
            ++this->environmentState.numberOfCheckedConfigs;
            ++this->environmentState.numValidConfigs;
            newEntry.stamp = this->env_kernelData.highestStamp + this->environmentState.stamp++;

#ifdef Debug
            std::cout << "[launch] New config pushed. Checked configs: "
                      << this->environmentState.numberOfCheckedConfigs
                      << ", Valid configs: " << this->environmentState.numValidConfigs << ", Stamp: " << newEntry.stamp
                      << "\n";
#endif

            emptyTheQueue(std::forward<T_Args>(launchArgs)...);
#ifdef Debug
            std::cout << "[launch] Exiting launch function.\n";
#endif
        }


    private:
        template<typename... T_Args>
        void emptyTheQueue(T_Args&&... launchArgs)
        {
            auto configWrapper = this->env_config_queue.get();
            if(configWrapper.has_value())
            {
                auto& config = configWrapper.value().get();
#ifdef Debug
                std::cout << "[emptyTheQueue] Applying config: " << config.config.toString() << "\n";
#endif
                applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config);
                return;
            }
#ifdef Debug
            std::cout << "queue is empty for the first time executing best Config " << "\n";
#endif
            executeBestConfig(std::forward<T_Args>(launchArgs)...);
            tune::benchmark::phaseAccessor(3);
            this->environmentState.sessionFinished = true;
        }

        template<typename... T_Args>
        bool handleFullQueue(T_Args&&... launchArgs)
        {
            auto configWrapper = this->env_config_queue.get();
            if(this->env_config_queue.full()
               || this->env_config_queue.size() >= this->environmentState.maxConfigsTotal)
            {
                auto& config = configWrapper.value().get();
#ifdef Debug
                std::cout << "[handleFullQueue] Queue full. Executing: " << config.config.toString() << "\n";
#endif
                applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config);
                return true;
            }
#ifdef Debug
            std::cout << "[handleFullQueue] Queue not full.\n";
#endif
            return false;
        }

        template<typename... T_Args>
        void executeBestConfig(T_Args&&... launchArgs)
        {
            auto& bestConfig = this->environmentState.getBestConfig();
            applyAndExecute(std::forward<T_Args>(launchArgs)..., bestConfig);
        }

        template<
            typename T_Queue,
            typename T_Exec,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_Kernelbundle,
            typename T_Config>
        void applyAndExecute(
            T_Queue&& queue,
            T_Exec&& exec,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
            T_Kernelbundle const& kernelbundle,
            T_Config& config)
        {
            auto& run = *this->env_kernelTuningPtr;
            run.fromConfig(config);
            applyCustomThreadSpec(run, spec);
            auto bundle = recreate(kernelbundle, run.m_userTuneables);

            trait::callPreProcessing(run, spec, this->env_metricInterface, bundle);

            using KernelFn = typename decltype(bundle)::KernelFn;

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

#define allowPrematureConfigSkip 1

        void update(auto& stored)
        {
            if(stored.state == ConfigState::Dummy)
                return;
            auto& run = *this->env_kernelTuningPtr;

            bool flagPre = stored.fullFlag;

#ifdef Debug
            std::cout << "[update] Pushing metric: " << run.metric << "\n";
#endif

            stored.pushMetric(run.metric);

            bool flagPost = stored.fullFlag;

#ifdef Debug
            std::cout << "  FullFlag (after): " << flagPost << "\n";
            std::cout << "  nr_runs (after): " << stored.nr_runs << "\n";
#endif

            if(!hasRunsPerConfig_Env())
            {
#ifdef Debug
                std::cout << " has no custom runs" << std::endl;
#endif
                if(flagPre != flagPost)
                {
#ifdef Debug
                    std::cout << "[update] Transitioned to full — counting as valid.\n";
#endif


                    detail::internal::assignBestIfBetter<typename Base::T_MetricInterfaceType>(
                        this->environmentState.getBestConfig(),
                        stored);
                    return;
                }
            }
            else
            {
                if(flagPre != flagPost)
                {
#ifdef Debug
                    std::cout << "[update] Flag changed — reset to false for early processing\n";
#endif
                    stored.fullFlag = false;
                }

                if(stored.nr_runs >= getRunsPerConfig())
                {
#ifdef Debug
                    std::cout << "[update] Reached max runs — setting fullFlag = true\n";
#endif
                    stored.fullFlag = true;
                    detail::internal::assignBestIfBetter<typename Base::T_MetricInterfaceType>(
                        this->environmentState.getBestConfig(),
                        stored);
                    return;
                }
                else
                {
#ifdef Debug
                    std::cout << "current runs " << stored.nr_runs << " vs demanded runs: " << getRunsPerConfig()
                              << "\n";
#endif
                }
            }

#if allowPrematureConfigSkip && defined(Debug)
            std::cout << "[update] Checking for premature skip of config\n";
#endif
            if constexpr(allowPrematureConfigSkip)
            {
                prematureConfigSkip(stored);
            }

#ifdef Debug
            std::cout << "[update] Finished with config:\n" << stored.toString() << "\n";
            std::cout << "  Final state: " << static_cast<int>(stored.state) << "\n";
            std::cout << "  FullFlag (final): " << stored.fullFlag << "\n";
#endif
        }

        template<typename T_Config>
        void prematureConfigSkip(ConfigEntry<T_Config>& stored)
        {
            auto& best = this->environmentState.getBestConfig();
            if(best == stored)
                return;
            if(stored.state != ConfigState::Initialized)
                return;
            auto res = best.compare(stored); // kruskal wallis comparison

            switch(res)
            {
            case ::Comparison::Greater:
                {
                    // best is higher then stored
                    auto& config = compareGetBest<typename Base::T_MetricInterfaceType>(best, stored);
                    if(best == config)
                    {
#ifdef Debug
                        std::cout << " this should never happen in a the timing scenario like this" << std::endl;
                        std::cout
                            << " best has a higher (worse) metric then stored. yet got returned by compareGetBest "
                            << std::endl;
#endif
                        ++this->environmentState.numberOfCheckedConfigs;
                        ++this->environmentState.numValidConfigs;
                        stored.fullFlag = true;
                    }
                }
            case ::Comparison::Less:
                {
                    // best is lower then stored
                    auto& config = compareGetBest<typename Base::T_MetricInterfaceType>(best, stored);
                    if(best == config)
                    {
                        // #ifdef Debug
                        std::cout << "[Config]" << "," << stored.toString() << "," << stored.getMedian() << std::endl;
                        std::cout << "[Best Config]" << "," << best.toString() << "," << best.getMedian() << std::endl;
                        // std::endl;
                        // #endif
                        ++this->environmentState.numberOfCheckedConfigs;
                        ++this->environmentState.numValidConfigs;
                        stored.fullFlag = true;
                    }
                }
            case ::Comparison::Inconclusive:
                {
                    break;
                }
            default:
                break;
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
    auto CTuneableBundle = alpaka::tune::trait::constructRuntimeCtuneablesForActivKernel(bundle);
    auto userTuple = extractTuneables(bundle);
    // Combine into kernel model
    auto completeRun = KernelTuningModel{userTuple, newRun.m_frameTuneables, CTuneableBundle};

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
    using model = KernelTuningModel<
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
        alpaka::core::demangledName(device),
        alpaka::core::demangledName(exec),
        alpaka::core::demangledName<T_KernelBundle>(),
        sessionSpecifier);
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
