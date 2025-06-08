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
    template<
        typename T_Device,
        typename T_Exec,
        typename T_NumFrames,
        typename T_FrameExtent,
        typename T_KernelBundle,
        typename T_Strategy,
        typename T_Interface,
        typename T_Constraint,
        typename T_Run,
        typename T_SessionSpecifier,
        typename T_History>
    auto setup_enqueue(

        T_Device device,
        T_Exec exec,
        alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
        T_KernelBundle const& kernelBundle,
        T_Strategy& strategy,
        T_Interface& metricInterface,
        T_Constraint& constraint,
        T_Run& run,
        T_SessionSpecifier& sessionSpecifier,
        T_History& history,
        std::string const& config)
    {
        if(!history.initialized)
        {
            history.initialized = true;
            if(!config.empty())
            {
                history.loadConfig(config);
            }
        }

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
                                     history)
                                     .get();

        if(sessionSpecifier != kernelptr->ptrToHistory->specifiers)
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
                            history)
                            .get();
        }
        using T_Context = decltype(kernelptr);
        return kernelptr;
    }

    template<typename T_MetricInterface>
    void assignBestIfBetter(StorageKernelRun& best, StorageKernelRun& stored)
    {
        assert(!stored.metricContainer.empty());
        if(best.metricContainer.empty() && !stored.metricContainer.empty())
        {
            best = stored;
            return;
        }
        bool storedIsSmaller
            = (stored.metricContainer.get(median_t{}).as<t_ns>() < best.metricContainer.get(median_t{}).as<t_ns>());
        if(storedIsSmaller)
        {
            best = aLTb<T_MetricInterface>{}(stored, best);
            return;
        }
        best = aGTb<T_MetricInterface>{}(stored, best);
    }

    template<typename T_KernelRun>
    void checkSessionFinishedCondition(T_KernelRun& run, EnvironmentState& state)
    {
        std::cout << state.maxConfigsTotal << " state total cofig " << state.numberOfCheckedConfigs
                  << " number of checked configs" << std::endl;
        if(hasMaxRuns_Env())
        {
            std::cout << state.numValidConfigs << " evaluated configs from " << state.maxValidEvaluations << std::endl;
        }

        if(state.numberOfCheckedConfigs >= state.maxConfigsTotal || state.numValidConfigs >= state.maxValidEvaluations)
        {
            state.sessionFinished = true;
        }
    }

#    define allowPrematureConfigSkip true

    template<typename T_MetricInterface, typename T_Config>
    bool enoughEvaluationsForConfig(T_Config& config, KernelData& data, EnvironmentState& environment)
    {
        auto runHash = config.toHash();
        if(!data.runs.contains(runHash))
        {
            data.runs[runHash] = toStore(config);
            return false;
        }

        StorageKernelRun& stored = data.runs[runHash];
        if(stored.fullFlag || stored.state == StorageKernelRun::State::Dummy)
        {
            return true;
        }

        if(stored.state != StorageKernelRun::State::Initialized || stored.metricContainer.empty())
        {
            return false;
        }

        if(stored.nr_runs >= getRunsPerConfig_Env())
        {
            if(!stored.fullFlag)
            {
                ++environment.numberOfCheckedConfigs;
                ++environment.numValidConfigs;
                stored.fullFlag = true;
            }

            return true;
        }

        StorageKernelRun& best = environment.bestConfig;

        if(best.toHash() == runHash)
        {
            if(!best.fullFlag) // this also catches cases where the first config (best by default) might be invalid due
                               // to constraints
            {
                return false;
            }
            return true;
        }
        auto res = best.compare(stored);

        switch(res)
        {
        case ::Comparison::Greater:
            {
                // best is higher then stored
                std::string bestHashTmp = best.toHash();
                best = aGTb<T_MetricInterface>{}(best, stored);
                if(bestHashTmp != best.toHash())
                {
                    if(allowPrematureConfigSkip && !stored.fullFlag)
                    {
                        ++environment.numberOfCheckedConfigs;
                        ++environment.numValidConfigs;
                        stored.fullFlag = true;
                        return true;
                    }
                }


                if(!stored.fullFlag)
                {
                    return false;
                }
                return true;
            }
        case ::Comparison::Less:
            {
                // best is lower then stored
                std::string bestHashTmp = best.toHash();
                best = aLTb<T_MetricInterface>{}(best, stored);
                if(bestHashTmp != best.toHash())
                {
                    if(allowPrematureConfigSkip && !stored.fullFlag)
                    {
                        ++environment.numberOfCheckedConfigs;
                        ++environment.numValidConfigs;
                        stored.fullFlag = true;
                        return true;
                    }
                }
                if(!stored.fullFlag)
                {
                    return false;
                }
                return true;
            }
        case ::Comparison::Inconclusive:
            {
                std::cout << " inconclusive " << std::endl;
                if(!stored.fullFlag)
                {
                    return false;
                }
                assignBestIfBetter<T_MetricInterface>(best, stored);
                return true;
            }
        default:
            break;
        }

        std::cout << " this should not be reachable " << std::endl;
        std::terminate();
    }

    template<typename T_Context, typename T_Constraints, typename Run, typename Data>
    bool violatesConstraint(Run& run, Data& data, T_Constraints& constraint, EnvironmentState& state)
    {
        auto runHash = run.toHash();
        auto const& stored = data.runs.at(runHash);
        if(stored.state == StorageKernelRun::State::Dummy)
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
            data.runs[runHash] = toStore(run);
            auto& stored = data.runs[runHash];
            using T_state = ALPAKA_TYPEOF(stored.state);
            ++state.numberOfCheckedConfigs;
            stored.metricContainer.clear();
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
        T_Config& config,
        KernelData& data,
        EnvironmentState& environment,
        T_Constraints& constraints)
    {
        bool enoughEvalutations = enoughEvaluationsForConfig<T_MetricInterface>(config, data, environment);

        bool violatesConstraint_ = violatesConstraint<T_Context>(config, data, constraints, environment);
        return !enoughEvalutations && !violatesConstraint_;
    }

#    define maxConsecutiveStrategyRuns 100

    // Check if a m_strategy should be applied and config should be skipped
    template<
        typename T_Context,
        typename Run,
        typename Data,
        typename SharedParams,
        typename Strategy,
        typename T_MetricInterface,
        typename T_Constraints>
    bool getNextValidConfig(
        Run& run,
        Data& data,
        SharedParams& sharedParams,
        Strategy& strategy,
        T_MetricInterface& metric_interface,
        EnvironmentState& environment,
        T_Constraints& constraints)
    {
        auto& stored = data.runs[run.toHash()];
        for(uint32_t numStrat = 0; numStrat < maxConsecutiveStrategyRuns; numStrat++)
        {
            std::string oldHash = run.toHash();
            strategy(metric_interface, sharedParams, run, data, environment);

            std::string newHash = run.toHash();

            if(newHash != oldHash
               && configReadyForRun<T_Context, T_MetricInterface>(run, data, environment, constraints))
            {
                return true;
            }
            if(environment.sessionFinished)
                break;
        }
        environment.sessionFinished = true;
        std::cout << " Did not find a suitable new config in " << maxConsecutiveStrategyRuns
                  << " iterations using the currently selected Strategy using best Config now: "
                  << environment.bestConfig.toHash() << std::endl;
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

    inline void applyConfigAndExecuteKernel(
        auto const& queue,
        auto exec,
        auto const& kernelBundle,
        auto& spec,
        tune::concepts::MetricInterface auto& interface,
        auto& run)
    {
        applyCustomThreadSpec(run, spec);
        auto bundle = recreate(kernelBundle, run.userTuneables);
        // static_assert(std::is_same_v<decltype(bundle), void()>);
        // we take the original KernelBundle here as userdefined traits are most likely according to the initial
        // KernelBundle Definition
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

    template<typename T_Context, typename T_MetricInterface>
    inline void updateMetrics(auto& run, KernelData& data, EnvironmentState& state)
    {
        auto runHash = run.toHash();
        using T_state = ALPAKA_TYPEOF(data.runs[runHash].state);
        std::cout << " ran config: " << runHash << " time " << run.metric << std::endl;
        StorageKernelRun& stored = data.runs[runHash];
        bool flagPre = stored.fullFlag;
        stored.pushMetric(run.metric);
        bool flagPost = stored.fullFlag;


        switch(stored.state)
        {
        case T_state::Uninitialized:
            stored.stamp = state.stamp++;
            stored.state = T_state::WarmUp;

            ++stored.warm_up_runs;
            break;
        case T_state::WarmUp:
            if(stored.warm_up_runs < WarmUpRuns)
            {
                ++stored.warm_up_runs;
            }
            else
            {
                stored.metricContainer.clear();
                stored.pushMetric(run.metric);
                stored.state = T_state::Initialized;
                stored.nr_runs = 1;
            }

            break;

        case T_state::Initialized:
            ++stored.nr_runs;
            break;

        default:
            break;
        }

        // update stopping criteria
        if(!hasRunsPerConfig_Env())
        {
            if(flagPre != flagPost)
            {
                ++state.numberOfCheckedConfigs;
                ++state.numValidConfigs;
                assignBestIfBetter<T_MetricInterface>(state.bestConfig, stored);
            }
        }
        else
        {
            if(stored.nr_runs >= getRunsPerConfig_Env())
            {
                if(!stored.fullFlag)
                {
                    ++state.numberOfCheckedConfigs;
                    ++state.numValidConfigs;
                }
                assignBestIfBetter<T_MetricInterface>(state.bestConfig, stored);
                stored.fullFlag = true;
            }
        }
    }
} // namespace alpaka::tune::detail::internal

namespace alpaka
{


    template<
        typename T_Config,
        typename T_Queue,
        typename T_Exec,
        typename T_KernelBundle,
        typename T_Spec,
        typename T_MetricInterface>
    void executeBestConfig(
        tune::TuningHistory& history,
        KernelData& data,
        EnvironmentState& state,
        std::string const& configfile,
        T_Config& config,
        T_Queue& queue,
        T_Exec& exec,
        T_KernelBundle& kernelBundle,
        T_Spec& spec,
        T_MetricInterface& metric_interface)
    {
        static bool write = true;
        StorageKernelRun& stored = data.runs[state.bestConfig.toHash()];
        if(write)
        {
            toActive(config, stored);
            history.storeConfig(configfile);
            write = false;
        }
        std::cout << " run with best config: " << config.toHash() << " median timings. "
                  << stored.metricContainer.get(median_t{}).template as<t_ns>() << std::endl;
        tune::detail::internal::applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metric_interface, config);
        onHost::wait(queue);
    }

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()

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


        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        tune::TuningHistory& history = tune::TuningHistory::get(config);
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;

        explicit TuningSession(
            T_Strategy strategy,
            T_MetricInterface interface,
            T_Constraints constraints,
            std::string config,
            std::size_t reRuns,
            std::size_t dynamicRuns,
            std::vector<std::string> sessionSpecifiers,
            KernelTuningModel<T_KernelRunArgs...> const& kernel_run)
            : m_strategy(std::move(strategy))
            , m_metricInterface(std::move(interface))
            , m_constraint(std::move(constraints))
            , config(std::move(config))
            , dynamicRuns_Nr(dynamicRuns)
            , sessionSpecifier(std::move(sessionSpecifiers))
            , m_run(kernel_run)
            , m_initialized(false)
        {
            history = tune::TuningHistory::get(this->config);
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
            onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
            T_KernelBundle const& kernelBundle)
        {
            auto* kernelptr = tune::detail::internal::setup_enqueue(
                device,
                exec,
                frameSpec,
                kernelBundle,
                this->m_strategy,
                this->m_metricInterface,
                this->m_constraint,
                this->m_run,
                sessionSpecifier,
                history,
                config);
            EnvironmentState& environment_state = kernelptr->environmentState;
            auto& activeRun = *kernelptr->activeRunPtr;
            auto& env_strategy = kernelptr->env_strategy;
            auto& env_metricInterface = kernelptr->env_metricInterface;
            auto& env_constraints = kernelptr->env_constraints;
            KernelData& historyKernelData = (*kernelptr->ptrToHistory);
            using T_Context = ALPAKA_TYPEOF(kernelptr);
            if(environment_state.sessionFinished)
            {
                executeBestConfig(
                    history,
                    historyKernelData,
                    environment_state,
                    config,
                    activeRun,
                    queue,
                    exec,
                    kernelBundle,
                    kernelptr->frameSpec,
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
                kernelptr->frameSpec,
                kernelptr->sharedParams);
        }

        //---internal_enqueue---
        template<
            typename T_Context,
            typename T_KernelBundle,
            typename T_kernelRun,
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
            KernelData& data,
            EnvironmentState& environment_state,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
            auto& sharedParameters)
        {
            using namespace alpaka::tune::detail::internal;
            if(configReadyForRun<T_Context, T_MetricInterface>(run, data, environment_state, constraints))
            {
                std::cout << " stopping criteria for current config:  " << run.toHash() << "not reached yet"
                          << std::endl;
                applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metricInterface, run);
                updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
                checkSessionFinishedCondition(run, environment_state);
                return;
            }
            bool foundNewConfig = getNextValidConfig<T_Context>(
                run,
                data,
                sharedParameters,
                strategy,
                metricInterface,
                environment_state,
                constraints);
            if(!foundNewConfig)
            {
                executeBestConfig(
                    history,
                    data,
                    environment_state,
                    config,
                    run,
                    queue,
                    exec,
                    kernelBundle,
                    spec,
                    metricInterface);
                return;
            }
            tune::detail::internal::applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metricInterface, run);
            alpaka::onHost::wait(queue);
            updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
            checkSessionFinishedCondition(run, environment_state);
        }

        ~TuningSession()
        {
            std::cout << " destructor called" << std::endl;
            std::cout << " init: " << m_initialized << std::endl;
            std::cout << " config: " << config << std::endl;
            if(history.initialized && config != "")
            {
                std::cout << " store Config called " << std::endl;
                history.storeConfig(config);
            }
        }
    };

    // KernelData stores Tuneable parameters as objects

    /*
     * is holding
     */

} // namespace alpaka


#endif // TUNER_H
#endif
