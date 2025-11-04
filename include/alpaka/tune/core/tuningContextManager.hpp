//
// Created by tim on 13.10.25.
//

#ifndef TUNINGCONTEXTMANAGER_H
#define TUNINGCONTEXTMANAGER_H
#include <alpaka/tune/core/peripherals/updateMetric.hpp>
#include <alpaka/tune/core/strategyContext.hpp>
#include <alpaka/tune/core/tuningContext.hpp>
#define BestMeasurements 10

namespace alpaka::tune
{

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
#ifdef Debug

    template<typename TConfig>
    std::string printConfigRecord(config::ConfigRecord<TConfig> const& config)
    {
        std::ostringstream oss;
        oss << '{';
        for(std::size_t i = 0; i < config.m_config.size(); ++i)
        {
            oss << config.m_config[i];
            if(i + 1 < config.m_config.size())
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
        using Base::Base; // inherit constructor from TuningContext

        bool writtenPersistent = false;

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
                executeBestConfig(std::forward<T_Args>(launchArgs)...);

#ifdef Debug
                std::cout << "[launch] Session has finished. bestCoutner:" << bestCounter << "\n";
#endif
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
            // SELECT valid config via Strategy
            auto nextConfig = selectNextConfig();
#ifdef Debug
            std::cout << " after strategy selection A" << std::endl;
#endif
            if(this->env_environmentState.strategyCriteriaReached())
            {
#ifdef Debug
                std::cout << "[launch] Strategy criteria or global break reached after loop" << std::endl;
#endif
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }
            // has to be instantiated at that point
            auto entry = this->getHistory().getRecord(nextConfig);
            if(!entry)
            {
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }


            this->env_config_queue.push_back(entry.value().get()); // insert valid + new config entry to queue


#ifdef Debug
            std::cout << "[launch] New config pushed. Checked configs: "
                      << this->env_environmentState.numberOfCheckedConfigs
                      << ", Valid configs: " << this->env_environmentState.numValidConfigs
                      << ", Stamp: " << entry.value().get().stamp << "\n";
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
            if(!writtenPersistent && !this->env_environmentState.sessionFinished)
            {
                if(!this->env_persistentHistory.m_filename.empty())
                    this->env_persistentHistory.write(
                        this->env_tuningModel,
                        this->env_activeHistory,
                        this->env_metaData);

                writtenPersistent = true;
                this->env_environmentState.sessionFinished = true;
            }
            executeBestConfig(std::forward<T_Args>(launchArgs)...);
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
                update(config, metric); // Update Metric
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
            auto const& bestConfig = this->env_environmentState.getBestConfig();
            if(bestConfig.has_value())
            {
                applyAndExecute(std::forward<T_Args>(launchArgs)..., bestConfig.value().get());
            }
            else
            {
                throw std::runtime_error("Trying to run a best configuration, without any prior tuning runs.");
            }
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
            this->env_tuningModel.applyToFrameSpec(currentSpec, configRecord.m_config);
            auto argsFromUserTunables = this->env_tuningModel.getValuesForRuntimeTuneables(configRecord.m_config);
            auto bundle = detail::recreate(kernelbundle, argsFromUserTunables);

            trait::callPreProcessing(this->env_tuningModel, currentSpec, this->env_metricInterface, bundle);

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
                auto indicies = this->env_tuningModel.getConfigSubset_CompileTuneables(configRecord.m_config);
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
            trait::callPostProcessing(this->env_tuningModel, currentSpec, this->env_metricInterface, bundle);
            return metric;
        }

        [[nodiscard]] auto selectNextConfig(uint32_t numAttempts = 0)
        {
            auto view = ConfigDescriptor{this->env_tuningModel};
            using T_View = decltype(view);
            using T_Config = decltype(T_View::getEmptyConfig());
            using T_Normalized = decltype(T_View::getEmptyNormalizedConfig());

            T_Config currentConfig = T_View::getEmptyConfig();
            auto ctx
                = alpaka::tune::StrategyContext<decltype(this->env_tuningModel), decltype(this->env_metricInterface)>{
                    view,
                    this->getHistory(),
                    this->env_environmentState};


            // run strategy
            auto config = this->env_strategy(ctx);
            using configType = decltype(config);
            static_assert(
                std::is_convertible_v<T_Normalized, configType> || std::is_convertible_v<T_Config, configType>,
                "Strategy has to return a Config!");

            if constexpr(std::is_same_v<std::remove_cvref_t<T_Config>, std::remove_cvref_t<configType>>)
            {
                currentConfig = std::move(config);
            }
            else
            {
                currentConfig = std::move(this->env_tuningModel.createConfigFromNormalized(config));
            }

#ifdef Debug
            std::cout << " retrieved config from strategy: " << printConfig(currentConfig) << std::endl;
#endif

            if(this->env_environmentState.strategyCriteriaReached(numAttempts))
            {
#ifdef Debug
                std::cout << " strate break criteria reached " << std::endl;
#endif
                this->env_environmentState.strategyFinished = true;
                return this->env_environmentState.getBestConfig().value().get().m_config;
            }

            config::ConfigRecord<decltype(currentConfig)>& configRecord
                = this->getHistory().getOrCreate(currentConfig);
            bool newConfig = (configRecord.state == config::ConfigState::Uninitialized);
            if(newConfig)
            {
                std::cout << " print new Config: " << printConfig(currentConfig) << std::endl;
                ++this->env_environmentState.numberOfCheckedConfigs;
                configRecord.state = config::ConfigState::Empty;
            }
            if(Base::violatesConstraint(configRecord))
                return selectNextConfig(++numAttempts);
            if(newConfig)
                ++this->env_environmentState.numValidConfigs;
            return currentConfig;
        }

#define allowPrematureConfigSkip 0

        template<concepts::ConfigLike T_Config>
        void update(config::ConfigRecord<T_Config>& stored, double_t const& metric)
        {
            static constexpr bool skip = allowPrematureConfigSkip; //@TODO get this from the context
            core::peripherals::updateMetrics<skip, typename Base::T_MetricInterfaceType, T_Config>(
                stored,
                this->env_environmentState,
                metric);
        }
    };

} // namespace alpaka::tune
#endif // TUNINGCONTEXTMANAGER_H
