//
// Created by tim on 13.10.25.
//

#ifndef TUNINGCONTEXTMANAGER_H
#define TUNINGCONTEXTMANAGER_H
#include <alpaka/tune/core/peripherals/updateMetric.hpp>
#include <alpaka/tune/core/strategyContext.hpp>
#include <alpaka/tune/core/tuningContext.hpp>
#define BestMeasurements 10
#define Debug

namespace alpaka::tune
{
#ifdef Debug
    template<typename TConfig>
    std::string printConfig(TConfig const& config)
    {
        std::ostringstream oss;
        oss << '{';
        for(std::size_t i = 0; i < config.size(); ++i)
        {
            oss << config[i];
            if(i + 1 < config.size())
                oss << ", ";
        }
        oss << '}';
        return oss.str();
    }

    template<typename TConfig>
    std::string printConfigRecord(config::ConfigRecord<TConfig> const& config)
    {
        std::ostringstream oss;
        oss << '{';
        for(std::size_t i = 0; i < config.config.size(); ++i)
        {
            oss << config.config[i];
            if(i + 1 < config.config.size())
                oss << ", ";
        }
        oss << '}';
        return oss.str();
    }

    template<typename T_Config>
    std::string printEnvironmentState(tune::core::peripherals::EnvironmentState<T_Config> const& env)
    {
        std::ostringstream oss;

        oss << "EnvironmentState {\n"
            << "  sessionFinished: " << std::boolalpha << env.sessionFinished << '\n'
            << "  strategyFinished: " << std::boolalpha << env.strategyFinished << '\n'
            << "  numberOfCheckedConfigs: " << env.numberOfCheckedConfigs << '\n'
            << "  numValidConfigs: " << env.numValidConfigs << '\n'
            << "  maxValidEvaluations: " << env.maxValidEvaluations << '\n'
            << "  maxConfigsTotal: " << env.maxConfigsTotal << '\n'
            << "  stamp: " << env.stamp << '\n'
            << "  strategyLimit: " << env.strategyLimit << '\n';

        if(env.bestConfig.has_value())
        {
            auto const& best = env.bestConfig->get();
            oss << "  bestConfig: ConfigRecord {\n";

            if constexpr(requires { best.getConfig(); })
            {
                auto const& cfg = best.getConfig();
                oss << "    config = {";
                for(std::size_t i = 0; i < cfg.size(); ++i)
                {
                    oss << cfg[i];
                    if(i + 1 < cfg.size())
                        oss << ", ";
                }
                oss << "}\n";
            }

            if constexpr(requires { best.getMeasurements(); })
            {
                auto const& measurements = best.getMeasurements();
                oss << "    numMeasurements = " << measurements.size() << '\n';
            }

            if constexpr(requires { best.fullFlag; })
            {
                oss << "    fullFlag = " << std::boolalpha << best.fullFlag << '\n';
            }

            oss << "  }\n";
        }
        else
        {
            oss << "  bestConfig: <none>\n";
        }

        oss << "}";
        return oss.str();
    }
#endif
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
            std::cout << "[Num evaluations]" << "," << this->env_environmentState.numValidConfigs << "\n";
            std::cout << "environment: " << printEnvironmentState(this->env_environmentState) << std::endl;
#endif

            if(this->env_environmentState.sessionFinished)
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
                    executeBestConfig(std::forward<T_Args>(launchArgs)...);
                    bestCounter++;
                    return;
                }
#ifdef Debug
                std::cout << "[launch] Executing best config again (count " << bestCounter << ").\n";
#endif
                executeBestConfig(std::forward<T_Args>(launchArgs)...);
                bestCounter++;
                return;
            }


            if(this->env_environmentState.globalBreakCriteriaFinished())
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
            auto view = ConfigDescriptor{this->env_kernelTuning};
            using T_view = decltype(view);
            auto currentConfig = T_view::getEmptyConfig();
            using T_Config = decltype(currentConfig);
            using T_Normalized = decltype(T_view::getEmptyNormalizedConfig());
            int i = 0;
            do
            {
                auto ctx = alpaka::tune::StrategyContext<
                    decltype(this->env_kernelTuning),
                    decltype(this->env_metricInterface)>{view, this->getConfigStorage(), this->env_environmentState};
                // run strategy
                auto config = this->env_strategy(ctx);

                using configType = decltype(config);
                static_assert(
                    std::is_convertible_v<T_Normalized, configType> || std::is_convertible_v<T_Config, configType>,
                    " Strategy has to return a Config!");
                if constexpr(std::is_same_v<
                                 std::remove_cvref_t<decltype(currentConfig)>,
                                 std::remove_cvref_t<decltype(config)>>)
                {
                    currentConfig = std::move(config);
                }
                else
                {
                    currentConfig = std::move(this->env_kernelTuning.createConfigFromNormalized(config));
                }
#ifdef Debug
                std::cout << " retrieved config from strategy: " << printConfig(currentConfig) << std::endl;
#endif
            } while(!this->env_environmentState.strategyCriteriaReached(++i)
                    && !this->env_environmentState.globalBreakCriteriaFinished()
                    && (this->getConfigStorage().contains(currentConfig) || Base::violatesConstraint(currentConfig)));
            if(this->env_environmentState.strategyCriteriaReached()
               || this->env_environmentState.globalBreakCriteriaFinished())
            {
                // #ifdef Debug
                //                 std::cout << "[launch] Strategy criteria or global break reached after loop.
                //                 Emptying
                //                              queue.\n ";
                // #endif
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }

#ifdef Debug
            std::cout << "[launch] Pushing new config: " << printConfig(currentConfig) << "\n";
#endif
            auto& newEntry = this->getConfigStorage().getOrCreate(currentConfig);
            this->env_config_queue.push_back(newEntry); // intert valid + new config entry to queue
            ++this->env_environmentState.numberOfCheckedConfigs;
            ++this->env_environmentState.numValidConfigs;
            newEntry.stamp = this->env_kernelData.highestStamp + this->env_environmentState.stamp++;

#ifdef Debug
            std::cout << "[launch] New config pushed. Checked configs: "
                      << this->env_environmentState.numberOfCheckedConfigs
                      << ", Valid configs: " << this->env_environmentState.numValidConfigs
                      << ", Stamp: " << newEntry.stamp << "\n";
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
                std::cout << "[emptyTheQueue] Applying config: " << printConfigRecord(config) << "\n";
#endif
                double_t metric = applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config, metric); // update Metric
                return;
            }
#ifdef Debug
            std::cout << "queue is empty for the first time executing best Config " << "\n";
#endif
            executeBestConfig(std::forward<T_Args>(launchArgs)...);
            this->env_environmentState.sessionFinished = true;
        }

        template<typename... T_Args>
        bool handleFullQueue(T_Args&&... launchArgs)
        {
            auto configWrapper = this->env_config_queue.get();
            if(this->env_config_queue.full()
               || this->env_config_queue.size() >= this->env_environmentState.maxConfigsTotal)
            {
                auto& config = configWrapper.value().get();
#ifdef Debug
                std::cout << "[handleFullQueue] Queue full. Executing: " << printConfigRecord(config) << "\n";
#endif
                double_t metric = applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config, metric); // Updat Metric
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
            auto& bestConfig = this->env_environmentState.getBestConfig();
            applyAndExecute(std::forward<T_Args>(launchArgs)..., bestConfig);
        }

        template<
            typename T_Queue,
            typename T_Exec,
            typename T_FrameSpecTuningModel,
            typename T_Kernelbundle,
            typename T_Config>
        double_t applyAndExecute(
            T_Queue const& queue,
            T_Exec&& exec,
            T_FrameSpecTuningModel const& specModel,
            T_Kernelbundle const& kernelbundle,
            config::ConfigRecord<T_Config> const& configRecord)
        {
            static auto currentSpec = alpaka::onHost::FrameSpec{
                specModel.m_spec.m_numFrames,
                specModel.m_spec.m_frameExtent,
                specModel.m_spec.m_threadSpec.m_numBlocks,
                specModel.m_spec.m_threadSpec.m_numThreads};
            this->env_kernelTuning.applyToFrameSpec(currentSpec, configRecord.config);
            auto argsFromUserTunables = this->env_kernelTuning.getValuesForRuntimeTuneables(configRecord.config);
            auto bundle = detail::recreate(kernelbundle, argsFromUserTunables);

            trait::callPreProcessing(this->env_kernelTuning, currentSpec, this->env_metricInterface, bundle);

            using KernelFn = typename decltype(bundle)::KernelFn;
            double_t metric;
            if constexpr(!trait::hasUserDefinedCTuneable<KernelFn>::value)
            {
                this->env_metricInterface.start();
                queue.enqueue(exec, currentSpec, bundle);
                onHost::wait(queue);
                metric = this->env_metricInterface.end();
            }
            else
            {
                auto indicies = this->env_kernelTuning.getConfigSubset_CompileTuneables(configRecord.config);
                alpaka::tune::CompileTimeHelpers::runtime_Kernel_dispatch<KernelFn>(
                    indicies,
                    [&](auto&& element)
                    {
                        auto newBundle = alpaka::apply(
                            [&element]<typename... T0>(T0&&... args)
                            { return KernelBundle{element, std::forward<T0>(args)...}; },
                            bundle.m_args);

                        this->env_metricInterface.start();
                        queue.enqueue(exec, currentSpec, newBundle);
                        onHost::wait(queue);
                        metric = this->env_metricInterface.end();
                    });
            }
            trait::callPostProcessing(this->env_kernelTuning, currentSpec, this->env_metricInterface, bundle);
            return metric;
        }

#define allowPrematureConfigSkip 1

        template<concepts::ConfigLike T_Config>
        void update(config::ConfigRecord<T_Config>& stored, double_t const& metric)
        {
            if(stored.state == config::ConfigState::Invalid)
                return;

            bool flagPre = stored.fullFlag;


            stored.pushMetric(metric); // copy metric to storage container

            bool flagPost = stored.fullFlag;

#ifdef Debug
            std::cout << "pushing time: " << metric << std::endl;
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
                    this->env_environmentState.template updateBestConfig<typename Base::T_MetricInterfaceType>(stored);
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
                    this->env_environmentState.template updateBestConfig<typename Base::T_MetricInterfaceType>(stored);
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
            std::cout << "[update] Finished with config:\n" << printConfigRecord(stored) << "\n";
            std::cout << "  Final state: " << static_cast<int>(stored.state) << "\n";
            std::cout << "  FullFlag (final): " << stored.fullFlag << "\n";
#endif
        }

        template<typename T_Config>
        void prematureConfigSkip(config::ConfigRecord<T_Config>& stored)
        {
            auto const& best = this->env_environmentState.getBestConfig();
            if(best == stored)
                return;
            if(stored.state != config::ConfigState::Initialized)
                return;
            auto res = best.compare(stored); // kruskal wallis comparison

            switch(res)
            {
            case alpaka::tune::config::Comparison::Greater:
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
                        ++this->env_environmentState.numberOfCheckedConfigs;
                        ++this->env_environmentState.numValidConfigs;
                        stored.fullFlag = true;
                    }
                }
            case config::Comparison::Less:
                {
                    // best is lower then stored
                    auto& config = compareGetBest<typename Base::T_MetricInterfaceType>(best, stored);
                    if(best == config)
                    {
#ifdef Debug
                        std::cout << "[Config]" << "," << printConfigRecord(stored) << "," << stored.getMedian()
                                  << std::endl;
                        std::cout << "[Best Config]" << "," << printConfigRecord(best) << "," << best.getMedian()
                                  << std::endl;
#endif
                        ++this->env_environmentState.numberOfCheckedConfigs;
                        ++this->env_environmentState.numValidConfigs;
                        stored.fullFlag = true;
                    }
                }
            case config::Comparison::Inconclusive:
                {
                    break;
                }
            default:
                break;
            }
        }
    };

} // namespace alpaka::tune
#endif // TUNINGCONTEXTMANAGER_H
