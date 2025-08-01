//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE
#include <alpaka/math/constants.hpp>
#include <alpaka/tune/active/constraint.hpp>
#include <alpaka/tune/utils/TimeEvent.hpp>
#include <alpaka/tune/utils/compileTimeTemplates.hpp>
#ifdef ENABLE_AUTOTUNE

#    include <alpaka/tune/active/sessionBuilder.h>

namespace alpaka

{
    bool anyTrue(alpaka::concepts::Vector auto const& vec)
    {
        bool result = false;
        for(auto i = 0; i < vec.dim(); ++i)
        {
            result = result || vec[i];
        }
        return result;
    }

    bool anyFalse(alpaka::concepts::Vector auto const& vec)
    {
        bool result = true;
        for(auto i = 0; i < vec.dim(); ++i)
        {
            result = result && !vec[i];
        }
        return result;
    }

    /*
     *converts a frameSpec to a KernelTuningModel
     *returns wether any part of the frameSpec was larger than the specified idxRange of the tuneable
     **/
    template<typename T_FrameSpec, typename... T_Args>
    static void addSpecToRun(KernelTuningModel<T_Args...>& kernelRun, T_FrameSpec& spec)
    {
        if constexpr(KernelTuningModel<T_Args...>::hasNumFramesTune())
        {
            kernelRun.getNumFramesTune().inputList.push_back(spec.m_numFrames);
        }
        if constexpr(KernelTuningModel<T_Args...>::hasFrameExtentTune())
        {
            kernelRun.getFrameExtentTune().inputList.push_back(spec.m_frameExtent);
        }
        if constexpr(KernelTuningModel<T_Args...>::hasNumBlocksTune())
        {
            kernelRun.getNumBlocksTune().inputList.push_back(spec.m_threadSpec.m_numBlocks);
        }
        if constexpr(KernelTuningModel<T_Args...>::hasThreadBlockSizeTune())
        {
            kernelRun.getThreadBlockSizeTune().inputList.push_back(spec.m_threadSpec.m_numThreads);
        }
    }

    template<typename T_FrameSpec, typename... T_Args>
    static T_FrameSpec& applyCustomThreadSpec(KernelTuningModel<T_Args...>& kernelRun, T_FrameSpec& spec)
    {
        if constexpr(KernelTuningModel<T_Args...>::hasNumFramesTune())
        {
            spec.m_numFrames = kernelRun.getNumFramesTune().value;
        }
        if constexpr(KernelTuningModel<T_Args...>::hasFrameExtentTune())
        {
            spec.m_frameExtent = kernelRun.getFrameExtentTune().value;
        }
        if constexpr(KernelTuningModel<T_Args...>::hasNumBlocksTune())
        {
            spec.m_threadSpec.m_numBlocks = kernelRun.getNumBlocksTune().value;
        }
        if constexpr(KernelTuningModel<T_Args...>::hasThreadBlockSizeTune())
        {
            spec.m_threadSpec.m_numThreads = kernelRun.getThreadBlockSizeTune().value;
        }

        return spec;
    }
} // namespace alpaka

namespace alpaka::tune::detail::internal
{
    template<typename T_MetricInterface, typename T_ConfigEntry>
    void assignBestIfBetter(T_ConfigEntry& best, T_ConfigEntry& stored)
    {
        assert(!stored.getMetrics().empty());

        if(best.getMetrics().empty() && !stored.getMetrics().empty())
        {
            best = stored;
            return;
        }
        if(!stored.fullFlag)
            return;
        best = compareGetBest<T_MetricInterface>(best, stored);
    }

#    define allowPrematureConfigSkip 0

    template<typename T_MetricInterface, typename T_Config>
    bool enoughEvaluationsForConfig(ConfigEntry<T_Config>& stored, EnvironmentState<T_Config>& environment)
    {
        if(stored.fullFlag || stored.state == ConfigState::Dummy)
        {
            return true;
        }
        if(stored.state != ConfigState::Initialized || stored.getMetrics().empty())
        {
            return false;
        }

        ConfigEntry<T_Config>& best = environment.bestConfig;

        if(best.toHash() == stored.toHash())
        {
            if(!best.fullFlag) // this also catches cases where the first config (best by default) might be invalid due
                               // to constraints
            {
                return false;
            }
            return true;
        }
        return false;
        if constexpr(allowPrematureConfigSkip)
        {
            auto res = best.compare(stored); // kruskal wallis comparison

            switch(res)
            {
            case ::Comparison::Greater:
                {
                    // best is higher then stored
                    std::string bestHashTmp = best.toHash();
                    best = compareGetBest<T_MetricInterface>(best, stored);
                    if(bestHashTmp == best.toHash())
                    {
                        ++environment.numberOfCheckedConfigs;
                        ++environment.numValidConfigs;
                        stored.fullFlag = true;
                        return true;
                    }
                    return false;
                }
            case ::Comparison::Less:
                {
                    // best is lower then stored
                    std::string bestHashTmp = best.toHash();
                    best = compareGetBest<T_MetricInterface>(best, stored);
                    if(bestHashTmp == best.toHash())
                    {
                        ++environment.numberOfCheckedConfigs;
                        ++environment.numValidConfigs;
                        stored.fullFlag = true;
                        return true;
                    }
                    return false;
                }
            case ::Comparison::Inconclusive:
                {
                    return false;
                }
            default:
                break;
            }
        }
    }

    template<typename T_Config>
    inline void checkSessionFinishedCondition(EnvironmentState<T_Config>& state)
    {
        // std::cout << state.maxConfigsTotal << " Total configs estimated " << state.numberOfCheckedConfigs
        //<< " number of checked configs" << std::endl;
        if(alpaka::tune::hasMaxRuns_Env())
        {
            std::cout << state.numValidConfigs << " evaluated configs from " << state.maxValidEvaluations << std::endl;
        }

        if(state.numberOfCheckedConfigs >= state.maxConfigsTotal || state.numValidConfigs >= state.maxValidEvaluations)
        {
            state.sessionFinished = true;
        }
    }

    template<
        typename T_MetricInterface,
        typename T_Device,
        typename T_Exec,
        typename T_NumFrames,
        typename T_FrameExtent,
        typename T_KernelBundle,
        typename T_Strategy,
        typename T_Interface,
        typename T_Constraint,
        typename T_Run,
        typename T_SessionSpecifier>
    auto* setup_enqueue(

        T_Device device,
        T_Exec exec,
        alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
        T_KernelBundle const& kernelBundle,
        T_Strategy& strategy,
        T_Interface& metricInterface,
        T_Constraint& constraint,
        T_Run& run,
        T_SessionSpecifier& sessionSpecifier,
        std::string const& config)
    {
        static auto* kernelptr = getTuningEnvironment(
                                     device,
                                     exec,
                                     frameSpec,
                                     kernelBundle,
                                     strategy,
                                     metricInterface,
                                     constraint,
                                     run,
                                     sessionSpecifier,
                                     config)
                                     .get();
        auto& data = kernelptr->env_kernelData;

        if(sessionSpecifier != data.specifiers)
        {
            kernelptr = getTuningEnvironment(
                            device,
                            exec,
                            frameSpec,
                            kernelBundle,
                            strategy,
                            metricInterface,
                            constraint,
                            run,
                            sessionSpecifier,
                            config)
                            .get();
        }
        if(!data.histEvaluated)
        {
            data.histEvaluated = true;
            auto& environment_state = kernelptr->environmentState;
            bool bestEvaluated
                = enoughEvaluationsForConfig<T_MetricInterface>(environment_state.bestConfig, environment_state)
                  && (environment_state.bestConfig.state != ConfigState::Dummy);
            for(auto& run : data.configEntries.getAll())
            {
                if(run.second.state == ConfigState::Dummy)
                    continue;
                auto& stored = run.second;

                bool enough = enoughEvaluationsForConfig<T_MetricInterface>(stored, environment_state);

                if(enough)
                {
                    if(bestEvaluated)
                    {
                        assignBestIfBetter<T_MetricInterface>(environment_state.bestConfig, stored);
                        continue;
                    }
                    environment_state.bestConfig = stored;
                }
                else
                {
                    kernelptr->env_config_queue.push_back(stored);
                }
            }
            checkSessionFinishedCondition(environment_state);
        }
        return kernelptr;
    }

    template<typename T_Context, typename Run, typename T_Constraints>
    bool violatesConstraint(Run& run, auto& configEntry, T_Constraints& constraint, auto& state)
    {
        auto& stored = configEntry;
        if(stored.state == ConfigState::Dummy)
        {
            return true;
        }
        bool valid = true;
        int index = 0;
        for_each(
            constraint,
            [&index, &run, &valid](auto& constraint)
            {
                auto constraintValid = constraint.template operator()<T_Context>(run);
                valid = valid && constraintValid;
            });
        if(!valid)
        {
            std::cout << " constraint violated for : " << configEntry.config.toString() << std::endl;
            using T_state = ALPAKA_TYPEOF(stored.state);
            ++state.numberOfCheckedConfigs;
            stored.getMetrics().clear();
            stored.stamp = -1;
            stored.state = T_state::Dummy;
            stored.fullFlag = true;
            stored.nr_runs = std::numeric_limits<decltype(stored.nr_runs)>::max();
            return true;
        }

        return false;
    }

    // Strategy functor and Constraint functor must be passed externally now
    template<typename T_Context, typename T_MetricInterface, typename T_Config, typename T_Constraints>
    bool configReadyForRun(
        auto& kernelTuningModel,
        T_Config& config,
        ConfigStorage<T_Config>& data,
        auto& environment,
        T_Constraints& constraints)
    {
        auto& configEntry = data.getOrCreate(config);
        bool violatesConstraint_
            = violatesConstraint<T_Context>(kernelTuningModel, configEntry, constraints, environment);
        if(violatesConstraint_)
            return false;
        bool enoughEvalutations = enoughEvaluationsForConfig<T_MetricInterface>(configEntry, environment);
        if(enoughEvalutations)
            return false;
        return true;
    }

#    define maxConsecutiveStrategyRuns 4000

    // Check if a m_strategy should be applied and config should be skipped
    template<
        typename T_Context,
        typename Run,
        typename T_Descriptor,
        typename T_Config,
        typename Strategy,
        typename T_MetricInterface,
        typename T_Constraints>
    bool getNextValidConfig(

        Run& run,
        KernelData<T_Config, T_Descriptor>& data,
        Strategy& strategy,
        T_MetricInterface& metric_interface,
        EnvironmentState<T_Config>& environment,
        T_Constraints& constraints)
    {
        std::cout << "[DEBUG] Starting getNextValidConfig()" << std::endl;
        std::cout << "[DEBUG] maxConsecutiveStrategyRuns: " << maxConsecutiveStrategyRuns << std::endl;
        std::cout << "[DEBUG] Initial sessionFinished: " << environment.sessionFinished << std::endl;

        for(uint32_t numStrat = 0; numStrat < maxConsecutiveStrategyRuns; numStrat++)
        {
            std::cout << "[DEBUG] Strategy iteration: " << numStrat + 1 << std::endl;

            auto oldConfig = run.toConfig();
            std::cout << "[DEBUG] Old Config: " << oldConfig.toString() << std::endl;
            std::cout << " [DEBUG] RUNNING STrategy: " << alpaka::core::demangledName<Strategy>() << std::endl;
            // Apply strategy
            auto modelView = KernelTuningModelView(run); // negligible overhead basically wraps the kernelTuningModel
            strategy(metric_interface, modelView, data.configEntries, environment);

            auto newConfig = run.toConfig();
            std::cout << "[DEBUG] New Config: " << newConfig.toString() << std::endl;

            if(oldConfig != newConfig)
            {
                std::cout << "[DEBUG] Config has changed." << std::endl;

                bool isReady = configReadyForRun<T_Context, T_MetricInterface>(
                    run,
                    newConfig,
                    data.configEntries,
                    environment,
                    constraints);

                std::cout << "[DEBUG] configReadyForRun returned: " << (isReady ? "true" : "false") << std::endl;

                if(isReady)
                {
                    std::cout << "[DEBUG] Found valid new config. Exiting." << std::endl;
                    return true;
                }
            }
            else
            {
                std::cout << "[DEBUG] Config did not change after strategy application." << std::endl;
            }

            if(environment.sessionFinished)
            {
                std::cout << "[DEBUG] Environment session is already marked finished. Exiting early." << std::endl;
                return false;
            }
        }

        environment.sessionFinished = true;

        auto& tuneableRange = std::get<0>(run.allTuneables()).idxRange;
        std::cout << "[DEBUG] No valid config found after all attempts." << std::endl;
        std::cout << "[DEBUG] Tuning Range - begin: " << tuneableRange.m_begin << ", end: " << tuneableRange.m_end
                  << ", stride: " << tuneableRange.m_stride << std::endl;

        std::cout << "[DEBUG] Using best known config instead: " << environment.bestConfig.toString() << std::endl;

        return false;
    }

    // Validate constraint or mark as Dummy


    template<typename T_KernelBundle>
    struct getTypeFrom
    {
    };

    template<typename KernelFn, typename... Args>
    struct getTypeFrom<KernelBundle<KernelFn, Args...>>
    {
        using type = KernelFn;
    };

    /*
     * seems like nvcc (CUDA/12.8.0) has this all or nothing approach on template type deduction,
     * either you specify all as
     * you would have pre cpp20
     * or you declare all with auto
     * -- otherwise most of the time you are having problems especially when using types that
     * contain variadic templates in their signature
     *
     */
    void applyConfigAndExecuteKernel(
        auto& config,
        auto const& queue,
        auto exec,
        auto const& kernelBundle,
        auto& spec,
        concepts::MetricInterface auto& interface,
        auto& run)
    {
        run.fromConfig(config); // apply config to tuningModell
        std::cout << "run with config: " << config.toString() << std::endl;
        applyCustomThreadSpec(run, spec); // apply frameSpecTunes to frameSpec
        auto bundle = recreate(kernelBundle, run.m_userTuneables); // apply runtime tuning parameter
        // static_assert(std::is_same_v<decltype(bundle), void()>);
        // we take the original KernelBundle here as userdefined traits are most likely according to the initial
        // KernelBundle Definition
        // optional preprocessing for example to write out dynamic shared memory
        trait::callPreProcessing(run, spec, interface, kernelBundle);

        using KernelFn = typename getTypeFrom<std::decay_t<decltype(kernelBundle)>>::type;
        if constexpr(!trait::hasUserDefinedCTuneable<KernelFn>::value)
        {
            interface.start(run, spec);
            queue.enqueue(exec, spec, bundle);
            onHost::wait(queue);
            interface.end(run, spec);
        }
        else
        {
            std::size_t i = trait::getRtimeIndexMap(kernelBundle)[run.compileTimeToFlatValueTuple()]; // kernelFn index
            static auto variants = typename trait::RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants{};
            alpaka::tune::runtime_Kernel_dispatch(
                i,
                variants,
                [&i, &bundle, &interface, &run, spec, &queue, exec](auto&& element)
                {
                    auto newBundle = std::apply(
                        [&element]<typename... T0>(T0&&... args)
                        { return KernelBundle{element, std::forward<T0>(args)...}; },
                        bundle.m_args);
                    interface.start(run, spec);
                    queue.enqueue(exec, spec, newBundle);
                    onHost::wait(queue);
                    interface.end(run, spec);
                });
        }
        trait::callPostProcessing(run, spec, interface, kernelBundle);

        // verifyCorrectness(NumNodes, spec.m_frameExtent);
    }

    template<typename NewKernelFn, typename OldKernelBundle>
    constexpr auto rebind_kernel(NewKernelFn&& newKernel, OldKernelBundle&& bundle)
    {
        return std::apply(
            [&](auto&&... args)
            {
                return KernelBundle<std::decay_t<NewKernelFn>, std::decay_t<decltype(args)>...>(
                    std::forward<NewKernelFn>(newKernel),
                    std::forward<decltype(args)>(args)...);
            },
            std::forward<OldKernelBundle>(bundle).m_args);
    }

#    define WarmUpRuns 1

    template<typename T_Context, typename T_MetricInterface, typename T_Config, typename T_Descriptor>
    inline void updateMetrics(auto& run, KernelData<T_Config, T_Descriptor>& data, EnvironmentState<T_Config>& state)
    {
        auto config = run.toConfig();
        ConfigEntry<T_Config>& stored = data.configEntries.getOrCreate(config);
        switch(stored.state)
        {
        case ConfigState::Uninitialized:
            stored.stamp = data.highestStamp + state.stamp++;
            break;
        case ConfigState::Dummy:
            return;
        default:
            break;
        }
        bool flagPre = stored.fullFlag;
        stored.pushMetric(run.metric);
        bool flagPost = stored.fullFlag;

        bool CIcriteriaReached = (flagPre != flagPost);
        bool customCriteriaReached = (stored.nr_runs >= alpaka::tune::getRunsPerConfig());
        // update stopping criteria
        if(!alpaka::tune::hasRunsPerConfig_Env())
        {
            if(CIcriteriaReached)
            {
                ++state.numberOfCheckedConfigs;
                ++state.numValidConfigs;
                assignBestIfBetter<T_MetricInterface>(state.bestConfig, stored);
            }
        }
        else
        {
            if(CIcriteriaReached)
            {
                stored.fullFlag = false;
            }
            if(customCriteriaReached)
            {
                ++state.numberOfCheckedConfigs;
                ++state.numValidConfigs;
                stored.fullFlag = true;
                assignBestIfBetter<T_MetricInterface>(state.bestConfig, stored);
            }
        }
    }
} // namespace alpaka::tune::detail::internal

namespace alpaka
{

#    define MeasureBestRuns 5 // how many runs after we have the best config will get messured (from the best config)

    template<
        typename T_Config,
        typename T_ConfigDescriptor,
        typename T_session,
        typename T_KernelTuningModel,
        typename T_Queue,
        typename T_Exec,
        typename T_KernelBundle,
        typename T_Spec,
        typename T_MetricInterface>
    void executeBestConfig(
        tune::TuningHistory& history,
        KernelData<T_Config, T_ConfigDescriptor>& kernel_data,
        EnvironmentState<T_Config>& state,
        T_session& session,
        T_KernelTuningModel& kernel_tuning_model,
        T_Queue& queue,
        T_Exec& exec,
        T_KernelBundle& kernelBundle,
        T_Spec& spec,
        T_MetricInterface& metric_interface)
    {
        static int runCount = 0;
        static bool write = true;

        ConfigEntry<T_Config>& stored = kernel_data.configEntries.getOrCreate(state.bestConfig.getConfig());
        kernel_tuning_model.fromConfig(stored.getConfig());
        auto currentConfig = kernel_tuning_model.toConfig();
        tune::detail::internal::applyConfigAndExecuteKernel(
            currentConfig,
            queue,
            exec,
            kernelBundle,
            spec,
            metric_interface,
            kernel_tuning_model);
        onHost::wait(queue);

        if(runCount < MeasureBestRuns)
        {
            stored.pushMetric(kernel_tuning_model.metric);
            ++runCount;

            if(write && runCount == MeasureBestRuns)
            {
                history.storeConfig(kernel_data);
                write = false;
                ++session.finishedConfigs;
                std::terminate();
            }
        }
    }

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()
    template<typename T>
    struct Dummy;

    template<
        typename T_Strategy = tune::strategy::randomSearch,
        typename T_MetricInterface = tune::metricInterface::Timing,
        typename T_Constraints = std::tuple<>,
        typename... T_KernelRunArgs>
    struct TuningSession
    {
        using T_floating = double_t;
        using T_Integer = std::size_t;

        T_Strategy m_strategy;
        T_MetricInterface m_metricInterface;
        T_Constraints m_constraint;
        KernelTuningModel<T_KernelRunArgs...> m_run;

        uint32_t finishedConfigs = 0;
        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;

        explicit TuningSession(
            T_Strategy&& strategy,
            T_MetricInterface interface,
            T_Constraints constraints,
            std::string config,
            std::size_t reRuns,
            std::size_t dynamicRuns,
            std::vector<std::string> sessionSpecifiers,
            KernelTuningModel<T_KernelRunArgs...> const& kernel_run)
            : m_strategy(std::forward<T_Strategy>(strategy))
            , m_metricInterface(std::move(interface))
            , m_constraint(std::move(constraints))
            , config(std::move(config))
            , dynamicRuns_Nr(dynamicRuns)
            , sessionSpecifier(std::move(sessionSpecifiers))
            , m_run(kernel_run)
        {
            this->reRuns = reRuns;
            m_run.metric = MetricUndefined;
        }

        template<typename... T_Specifiers>
        TuningSession& withRunSpecifiers(T_Specifiers... specifiers)
        {
            TuningSession neu = *this;

            processArgs(neu.sessionSpecifier, specifiers...);
            return *this;
        }

        /*
        TuningSession(const TuningSession&) = delete;
        TuningSession& operator=(const TuningSession&) = delete;
        TuningSession(TuningSession&&) noexcept = default;
        TuningSession& operator=(TuningSession&&) noexcept = default;*/
        template<typename Tuple, std::size_t... I>
        auto copyTupleImpl(Tuple const& t, std::index_sequence<I...>)
        {
            return std::make_tuple(std::get<I>(t)...);
        }

        template<typename... Ts>
        auto copyTuple(std::tuple<Ts...> const& t)
        {
            return copyTupleImpl(t, std::index_sequence_for<Ts...>{});
        }

        /** Enqueue and Execute a kernel for the tuning session
         * @param device
         * @param queue
         * @param exec
         * @param frameSpec
         * @param executor description how native worker threads will be mapped and grouped to compute grid layers
         * (blocks, threads).
         * @param specification thread or frame specification which provides a chunked description of the thread or
         * frame index domain
         * @param kernelBundle the compute kernel and there arguments
         */
        template<
            typename T_Device,
            typename T_Queue,
            typename T_Exec,
            typename T_NumFrames,
            typename T_FrameExtent,
            typename T_KernelBundle>
        auto enqueue(
            T_Device device,
            T_Queue& queue,
            T_Exec exec,
            onHost::FrameSpec<T_NumFrames, T_FrameExtent>& frameSpec,
            T_KernelBundle const& kernelBundle)
        {
            auto* kernelptr = tune::detail::internal::setup_enqueue<T_MetricInterface>(
                device,
                exec,
                frameSpec,
                kernelBundle,
                this->m_strategy,
                this->m_metricInterface,
                this->m_constraint,
                this->m_run,
                sessionSpecifier,
                config);

            auto& environment_state = kernelptr->environmentState;
            auto& activeRun = *kernelptr->env_kernelTuningPtr;
            auto& env_strategy = kernelptr->env_strategy;
            auto& env_metricInterface = kernelptr->env_metricInterface;
            auto& env_constraints = kernelptr->env_constraints;
            auto& historyKernelData = kernelptr->env_kernelData;
            using T_Context = ALPAKA_TYPEOF(kernelptr);
            if(environment_state.sessionFinished)
            {
                executeBestConfig(
                    tune::TuningHistory::get(config),
                    historyKernelData,
                    environment_state,
                    *this,
                    activeRun,
                    queue,
                    exec,
                    kernelBundle,
                    frameSpec,
                    env_metricInterface);
                return;
            }
            internal_enqueue<T_Context>(
                queue,
                exec,
                kernelBundle,
                env_strategy,
                env_metricInterface,
                env_constraints,
                activeRun,
                historyKernelData,
                environment_state,
                frameSpec,
                kernelptr->env_config_queue);
        }

        //---internal_enqueue---
        template<
            typename T_Context,
            typename T_KernelBundle,
            typename T_kernelRun,
            typename T_Config,
            typename T_ConfigDescriptor,
            typename T_NumBlocks,
            typename T_NumThreads>
        void internal_enqueue(
            auto const& queue,
            auto exec,
            T_KernelBundle const& kernelBundle,
            T_Strategy& strategy,
            T_MetricInterface& metricInterface,
            T_Constraints& constraints,
            T_kernelRun& run,
            KernelData<T_Config, T_ConfigDescriptor>& data,
            EnvironmentState<T_Config>& environment_state,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
            auto& config_queue)
        {
            using namespace alpaka::tune::detail::internal;

            std::cout << "[Session] Trying to fetch next config from round-robin queue...\n";
            auto res = config_queue.getRoundRobin();

            if(res.has_value())
            {
                std::cout << "[Session] Round-robin queue returned a config. Using current run state to map it.\n";

                auto initConfig = run.toConfig();
                std::cout << "[Session] Initial config (from run):\n" << initConfig.toString() << "\n";

                ConfigEntry<T_Config>& stored = data.configEntries.getOrCreate(initConfig);
                run.fromConfig(stored);

                std::cout << "[Session] run state updated from stored config entry.\n";
            }

            auto curConfig = run.toConfig();
            std::cout << "[Session] Current config (after queue/fallback):\n" << curConfig.toString() << "\n";

            if(configReadyForRun<T_Context, T_MetricInterface>(
                   run,
                   curConfig,
                   data.configEntries,
                   environment_state,
                   constraints))
            {
                std::cout << "[Session] Config is ready for execution (criteria not yet reached).\n";

                applyConfigAndExecuteKernel(curConfig, queue, exec, kernelBundle, spec, metricInterface, run);

                std::cout << "[Session] Kernel executed for current config.\n";

                updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
                std::cout << "[Session] Metrics updated for current config.\n";

                checkSessionFinishedCondition(environment_state);
                std::cout << "[Session] Session finished condition re-evaluated.\n";

                return;
            }

            std::cout << "[Session] Current config not ready. Attempting to generate next valid config...\n";

            bool foundNewConfig = tune::detail::internal::getNextValidConfig<T_Context>(
                run,
                data,
                strategy,
                metricInterface,
                environment_state,
                constraints);

            if(!foundNewConfig)
            {
                std::cout << "[Session] No new valid config found. Falling back to best config execution.\n";

                executeBestConfig(
                    tune::TuningHistory::get(config),
                    data,
                    environment_state,
                    *this,
                    run,
                    queue,
                    exec,
                    kernelBundle,
                    spec,
                    metricInterface);

                std::cout << "[Session] Best config executed. Ending iteration.\n";
                return;
            }

            auto config = run.toConfig();
            std::cout << "[Session] New config generated:\n" << config.toString() << "\n";

            tune::detail::internal::applyConfigAndExecuteKernel(
                config,
                queue,
                exec,
                kernelBundle,
                spec,
                metricInterface,
                run);

            alpaka::onHost::wait(queue);
            std::cout << "[Session] Kernel executed and host wait complete.\n";

            updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
            std::cout << "[Session] Metrics updated for new config.\n";

            checkSessionFinishedCondition(environment_state);
            std::cout << "[Session] Session finished condition checked.\n";
        }

        ~TuningSession()
        {
            std::cout << " destructor called" << std::endl;
        }
    };

    // KernelData stores Tuneable parameters as objects

    /*
     * is holding
     */

} // namespace alpaka


#endif // TUNER_H
#endif
