//
// Created by tim on 13.10.25.
//

#ifndef TUNINGCONTEXTMANAGER_H
#define TUNINGCONTEXTMANAGER_H

#define BestMeasurements 10

namespace alpaka::tune
{
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
            std::cout << "[Num evaluations]" << "," << this->environmentState.numValidConfigs << "\n";

#endif

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
        template<typename Dummy>
        struct Humb;

        template<
            typename T_Queue,
            typename T_Exec,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_ThreadSpec,
            typename T_Kernelbundle,
            typename T_Config>
        void applyAndExecute(
            T_Queue&& queue,
            T_Exec&& exec,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads, T_ThreadSpec>& spec,
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
                        auto newBundle = alpaka::apply(
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
#ifdef Debug
                        std::cout << "[Config]" << "," << stored.toString() << "," << stored.getMedian() << std::endl;
                        std::cout << "[Best Config]" << "," << best.toString() << "," << best.getMedian() << std::endl;
#endif
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
#endif // TUNINGCONTEXTMANAGER_H
