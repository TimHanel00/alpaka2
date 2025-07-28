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
    template<typename T_MetricInterface>
    void assignBestIfBetter(StorageKernelRun& best, StorageKernelRun& stored)
    {
        assert(!stored.metricContainer.empty());
        if(best.metricContainer.empty() && !stored.metricContainer.empty())
        {
            best = stored;
            return;
        }
        if(!best.fullFlag || !stored.fullFlag) // not yet fully evaluated both configs
            return;
        bool storedIsSmaller
            = (stored.metricContainer.get(median_t{}).as<t_ns>() < best.metricContainer.get(median_t{}).as<t_ns>());
        if(storedIsSmaller)
        {
            best = aLTb<T_MetricInterface>{}(stored, best);
            return;
        }
        best = aGTb<T_MetricInterface>{}(stored, best);
    }

#    define allowPrematureConfigSkip false

    template<typename T_MetricInterface>
    bool enoughEvaluationsForConfig(StorageKernelRun& stored, EnvironmentState& environment)
    {
        if(stored.fullFlag || stored.state == StorageKernelRun::State::Dummy)
        {
            return true;
        }
        if(stored.state != StorageKernelRun::State::Initialized || stored.metricContainer.empty())
        {
            return false;
        }

        StorageKernelRun& best = environment.bestConfig;

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
        if(allowPrematureConfigSkip)
        {
            auto res = best.compare(stored); // kruskal wallis comparison

            switch(res)
            {
            case ::Comparison::Greater:
                {
                    // best is higher then stored
                    std::string bestHashTmp = best.toHash();
                    best = aGTb<T_MetricInterface>{}(best, stored);
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
                    best = aLTb<T_MetricInterface>{}(best, stored);
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

    inline void checkSessionFinishedCondition(EnvironmentState& state)
    {
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
        KernelData& data = *kernelptr->ptrToHistory;

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
                            history)
                            .get();
        }
        if(!data.histEvaluated)
        {
            data.histEvaluated = true;
            EnvironmentState& environment_state = kernelptr->environmentState;
            bool bestEvaluated
                = enoughEvaluationsForConfig<T_MetricInterface>(environment_state.bestConfig, environment_state)
                  && (environment_state.bestConfig.state != StorageKernelRun::State::Dummy);
            for(auto& run : data.runs)
            {
                if(run.second.state == StorageKernelRun::State::Dummy)
                {
                    ++environment_state.numberOfCheckedConfigs;
                    continue;
                }
                if(run.second.fullFlag)
                {
                    ++environment_state.numberOfCheckedConfigs;
                    ++environment_state.numValidConfigs;
                    if(!bestEvaluated)
                    {
                        environment_state.bestConfig = run.second;
                        continue;
                    }
                    assignBestIfBetter<T_MetricInterface>(environment_state.bestConfig, run.second);


                    continue;
                }
                if(enoughEvaluationsForConfig<T_MetricInterface>(run.second, environment_state))
                {
                    if(!bestEvaluated)
                    {
                        environment_state.bestConfig = run.second;
                        continue;
                    }
                    assignBestIfBetter<T_MetricInterface>(environment_state.bestConfig, run.second);


                    continue;
                }
                environment_state.config_queue.push_back(run.second);
            }
            checkSessionFinishedCondition(environment_state);
        }
        using T_Context = decltype(kernelptr);
        return kernelptr;
    }

    template<typename T_Context, typename T_Constraints, typename Run, typename Data>
    bool violatesConstraint(Run& run, Data& data, T_Constraints& constraint, EnvironmentState& state)
    {
        auto runHash = run.toHash();
        auto& stored = data.runs.at(runHash);
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
            // std::cout << " constraint violated for : " << runHash << std::endl;
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
        if(!data.runs.contains(config.toHash()))
        {
            data.runs[config.toHash()] = toStore(config);
            bool violatesConstraint_ = violatesConstraint<T_Context>(config, data, constraints, environment);
            if(violatesConstraint_)
                return false;
            return true;
        }
        StorageKernelRun& stored = data.runs[config.toHash()];
        bool violatesConstraint_ = violatesConstraint<T_Context>(config, data, constraints, environment);
        if(violatesConstraint_)
            return false; // shortcut long evaluation
        bool enoughEvalutations = enoughEvaluationsForConfig<T_MetricInterface>(stored, environment);
        if(enoughEvalutations)
            return false;
        return true;
    }

#    define maxConsecutiveStrategyRuns 40000

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
            {
                std::cout << " environment is already finished " << std::endl;
                return false;
            }
        }
        environment.sessionFinished = true;
        // KernelTuningModel<> run;
        auto& tuneableRange = std::get<0>(run.allTuneables()).idxRange;
        std::cout << " begin: " << tuneableRange.m_begin << " end: " << tuneableRange.m_end
                  << " stride: " << tuneableRange.m_stride << std::endl;
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

    template<typename numBlocks, typename numThreads, typename... Args>
    struct ConfigContext
    {
        onHost::FrameSpec<numBlocks, numThreads> spec;
        KernelBundle<Args...> kernelBundle;
        ConfigContext(onHost::FrameSpec<numBlocks, numThreads> const& spec, KernelBundle<Args...> const& bundle)
            : spec(spec)
            , kernelBundle(bundle) {};

        auto getFrameSpec()
        {
            return spec;
        };

        auto getKernelBundle()
        {
            return kernelBundle;
        };
    };

    auto getbestHelper(auto const& kernelBundle, auto& spec, auto& run)
    {
        applyCustomThreadSpec(run, spec);
        auto bundle = recreate(kernelBundle, run.userTuneables);
        // static_assert(std::is_same_v<decltype(bundle), void()>);
        // we take the original KernelBundle here as userdefined traits are most likely according to the initial
        // KernelBundle Definition
        // std::cout << "launching Kernel: " << run.toHash() << std::endl;
        using KernelFn = typename getTypeFrom<std::decay_t<decltype(kernelBundle)>>::type;

        if constexpr(!trait::hasUserDefinedCTuneable<KernelFn>::value)
        {
            return ConfigContext{spec, bundle};
        }
        else
        {
            std::size_t i = trait::getRtimeIndexMap(kernelBundle)[run.compileTimeToFlatValueTuple()]; // kernelFn index
            static auto variants = typename trait::RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants{};
            alpaka::tune::runtime_Kernel_dispatch(
                i,
                variants,
                [&i, &bundle, spec](auto&& element)
                {
                    auto newBundle = std::apply(
                        [&element]<typename... T0>(T0&&... args)
                        { return KernelBundle{element, std::forward<T0>(args)...}; },
                        bundle.m_args);
                    return ConfigContext{spec, newBundle};
                });
        }
    }

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
        // std::cout << "launching Kernel: " << run.toHash() << std::endl;
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
        StorageKernelRun& stored = data.runs[runHash];
        switch(stored.state)
        {
        case T_state::Uninitialized:
            stored.stamp = data.highestStamp + state.stamp++;
            break;
        case T_state::Dummy:
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

#    define MeasureBestRuns                                                                                           \
        1000 // how many runs after we have the best config will get messured (from the best config)

    template<typename... T_Args>
    void defaultSIMD(auto& queue, const auto& defaultSpec, KernelTuningModel<T_Args...>& config)
    {
        auto elementsPerFrameItem = getNumElemPerThread<float_t>(queue);
        for_each(
            config.allTuneables(),
            [&](auto& tune)
            {
                if(tune.name() == "CTune_0")
                {
                    tune.value = elementsPerFrameItem;
                }
            });
    }

    template<typename T_KernelBundle>
    struct configStore
    {
        std::string defaultHash;
        std::string configHash;
        std::string const configFile = "best_default_summary.txt";
        std::string demangled = "";
        configStore() = default;
        configStore(std::string const& configHash, std::string const& defaultHash, std::string const& bundle)
            : configHash(configHash)
            , defaultHash(defaultHash)
            , demangled(bundle) {

            };
        configStore(configStore const&) = default;
        configStore& operator=(configStore const&) = default;

        void store()
        {
            std::ofstream out(configFile, std::ios::app); // append mode
            if(out)
            {
                out << "\n"; // spacing from previous entry

                // Optional: entry separator
                out << "==============================\n";

                // Timestamp
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                out << "Timestamp: " << std::put_time(std::localtime(&now_c), "%F %T") << "\n";

                // Config details
                out << "Demangled: " << demangled << "\n";
                out << "BestConfigID: " << configHash << "\n";
                out << "DefaultConfigID: " << defaultHash << "\n";

                out << "==============================\n";
            }
        }
    };
    template<typename Dummy>
    struct DummyH;

    void selectRun(
        auto& queue,
        auto& config,
        KernelData& data,
        auto const& defaultSpec,
        auto const& kernelbundle,
        int runCount,
        EnvironmentState& state)
    {
        if(static_cast<uint32_t>(runCount / 4) % 2 == 0)
        {
            // run Best
            std::cout << "Best" << std::endl;
            StorageKernelRun& best = data.runs[state.bestConfig.toHash()];
            toActive(config, best);
        }
        else
        {
            std::cout << "Default" << std::endl;
            using KernelFn = typename tune::detail::internal::getTypeFrom<std::decay_t<decltype(kernelbundle)>>::type;
            using Vec_2 = decltype(defaultSpec.m_numFrames);
            alpaka::tune::trait::getDefault<KernelFn, Vec_2>(queue, config);
            if(!data.runs.contains(config.toHash()))
            {
                data.runs[config.toHash()] = toStore(config);
            }
        }
    }

    template<
        typename T_session,
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
        T_session& session,
        T_Config& config,
        T_Queue& queue,
        T_Exec& exec,
        T_KernelBundle& kernelBundle,
        T_Spec& spec,
        T_Spec const& defaultSpec,
        T_MetricInterface& metric_interface)
    {
        static int runCount = 0;
        static int storedRuns = 0;
        static bool write = true;
        // StorageKernelRun& stored = data.runs[state.bestConfig.toHash()];
        selectRun(queue, config, data, defaultSpec, kernelBundle, runCount, state);
        tune::detail::internal::applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metric_interface, config);
        onHost::wait(queue);
        StorageKernelRun& cur = data.runs[config.toHash()];
        if(storedRuns < MeasureBestRuns * 2)
        {
            if(runCount %4 > 1)
            {
                cur.pushMetric(config.metric);
                storedRuns++;
            }
            ++runCount;
            if(write && runCount == MeasureBestRuns * 2)
            {
                {
                    std::string bestRun = state.bestConfig.toHash();
                    std::string defaultRun = config.toHash();
                    configStore<T_KernelBundle>{bestRun, defaultRun, core::demangledName(kernelBundle)}.store();
                }
                history.storeConfig(configfile);
                ++session.finishedConfigs;
            }
        }
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

        uint32_t finishedConfigs = 0;
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

        template<typename T_KernelModel, typename... T_Args>
        class TupleHolder
        {
        public:
            TupleHolder() = default;

            void addDefaults(T_Args&&... args)
            {
                data = std::make_tuple(std::forward<T_Args>(args)...);
            }

            template<std::size_t N>
            auto& get()
            {
                static_assert(N < sizeof...(T_Args), "Index out of bounds");
                return std::get<N>(data);
            }

            template<std::size_t N>
            auto const& get() const
            {
                static_assert(N < sizeof...(T_Args), "Index out of bounds");
                return std::get<N>(data);
            }

        private:
            std::tuple<T_Args...> data;
        };

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

        template<
            typename T_Device,
            typename T_Queue,
            typename T_Exec,
            typename T_NumFrames,
            typename T_FrameExtent,
            typename T_KernelBundle>
        auto getbest(
            T_Device device,
            T_Queue& queue,
            T_Exec exec,
            onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
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
                history,
                config);
            EnvironmentState& environment_state = kernelptr->environmentState;
            auto& activeRun = *kernelptr->activeRunPtr;
            toActive(activeRun, environment_state.bestConfig);
            return tune::detail::internal::getbestHelper(kernelBundle, kernelptr->frameSpec, activeRun);
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
                    *this,
                    activeRun,
                    queue,
                    exec,
                    kernelBundle,
                    kernelptr->frameSpec,
                    kernelptr->defaultFrameSpec,
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
                kernelptr->defaultFrameSpec,
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
            onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& defaultSpec,
            auto& sharedParameters)
        {
            using namespace alpaka::tune::detail::internal;
            auto res = environment_state.config_queue.getRoundRobin();
            if(res.has_value())
            {
                // this queue only reads  from a toml file it can be ignored for now
                StorageKernelRun& stored = res.value();
                toActive(run, stored);
            }
            if(configReadyForRun<T_Context, T_MetricInterface>(run, data, environment_state, constraints))
            {
                applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metricInterface, run);
                updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
                checkSessionFinishedCondition(environment_state);
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
                    *this,
                    run,
                    queue,
                    exec,
                    kernelBundle,
                    spec,
                    defaultSpec,
                    metricInterface);
                return;
            }
            tune::detail::internal::applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, metricInterface, run);
            alpaka::onHost::wait(queue);
            updateMetrics<T_Context, T_MetricInterface>(run, data, environment_state);
            checkSessionFinishedCondition(environment_state);
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
