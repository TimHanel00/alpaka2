//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE

#ifdef ENABLE_AUTOTUNE

#    include <alpaka/tune/active/sessionBuilder.h>

namespace alpaka
{
    template<typename T_FrameSpec, typename... T_Args>
    static T_FrameSpec& applyCustomThreadSpec(ActiveKernelRun<T_Args...>& kernelRun, T_FrameSpec& spec)
    {
        if constexpr(kernelRun.hasNumFramesTune())
        {
            spec.m_numFrames = kernelRun.getNumFramesTune().value;
        }
        if constexpr(kernelRun.hasFrameExtentTune())
        {
            spec.m_frameExtent = kernelRun.getFrameExtentTune().value;
        }
        if constexpr(kernelRun.hasNumBlocksTune())
        {
            spec.m_threadSpec.m_numBlocks = kernelRun.getNumBlocksTune().value;
        }
        if constexpr(kernelRun.hasThreadBlockSizeTune())
        {
            spec.m_threadSpec.m_numThreads = kernelRun.getThreadBlockSizeTune().value;
        }

        return spec;
    }

    template<typename... Args>
    void printGBFromActive(ActiveKernelRun<Args...>& active)
    {
        if(active.hasNumBlocksTune())
        {
            std::cout << "G: " << active.getNumBlocksTune().valueToString() << std::endl;
            return;
        }
        if(active.hasNumFramesTune())
        {
            std::cout << "B: " << active.threadBlockSize->valueToString() << std::endl;
            return;
        }

        if(active.threadBlockSize.has_value() && active.numBlocksTune.has_value())
        {
            std::cout << "G: " << active.numBlocksTune->valueToString()
                      << " , B: " << active.threadBlockSize->valueToString() << std::endl;
        }
    }

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()

    template<typename T_Strategy = alpaka::tune::strategy::randomSearch, typename... T_KernelRunArgs>
    struct TuningSession
    {
        using T_floating = double_t;
        using T_Integer = std::size_t;
        ActiveKernelRun<T_KernelRunArgs...> run;
        tune::TuningHistory& history = tune::TuningHistory::get();
        T_Strategy strategy;
        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;

        explicit TuningSession(
            T_Strategy strategy,
            std::string config,
            std::size_t reRuns,
            std::size_t dynamicRuns,
            std::vector<std::string> sessionSpecifiers,
            ActiveKernelRun<T_KernelRunArgs...> const& kernel_run)
            : strategy(std::move(strategy))
            , config(std::move(config))
            , dynamicRuns_Nr(dynamicRuns)
            , sessionSpecifier(std::move(sessionSpecifiers))
            , run(kernel_run)
            , m_initialized(false)
        {
            this->reRuns = getRunsPerConfig();
            run.metric = MetricUndefined;
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
            if(!history.initialized)
            {
                history.initialized = true;
                if(!config.empty())
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Loading history..." << std::endl;
#    endif
                    history.loadConfig(config);
                }
            }
#    ifdef DEBUG
            std::cout << "[DEBUG] After load history." << std::endl;
#    endif
            static auto kernelptr
                = createKernelSingleton(device, exec, frameSpec, kernelBundle, this->run, sessionSpecifier, history);

            auto& activeRun = *kernelptr->activeRunPtr;
            if(sessionSpecifier
               != kernelptr->ptrToHistory->specifiers) // super ugly but thats currently the solution to handle
                                                       // multiple tuning sessions with different sessionSpecifier but
                                                       // same template types
            {
                kernelptr = createKernelSingleton(
                    device,
                    exec,
                    frameSpec,
                    kernelBundle,
                    this->run,
                    sessionSpecifier,
                    history);
                activeRun = *kernelptr->activeRunPtr;
                activeRun.resetSignal = true;
            }
#    ifdef DEBUG
            std::cout << "[DEBUG] Kernel singleton created." << std::endl;
#    endif


            KernelData& historyKernelData = (*kernelptr->ptrToHistory);
#    ifdef DEBUG
            std::cout << "[DEBUG] kernelBundle demangled: " << alpaka::core::demangledName(kernelBundle) << std::endl;
#    endif

            if(historyKernelData.sumOfRuns >= getMaxRuns(activeRun.maxRuns) * this->reRuns)
            {
                std::cout << "[DEBUG] Selecting best config." << std::endl;
                auto event = tune::createTimeEventFromActive(activeRun);
                alpaka::tune::strategy::bestRecorded{}(activeRun, historyKernelData.runs);
                std::cout << activeRun.toHash() << std::endl;
                applyCustomThreadSpec(activeRun, kernelptr->frameSpec);
#    ifdef DEBUG
                std::cout << "[DEBUG] Best config applied: Blocks = " << kernelptr->frameSpec.m_threadSpec.m_numBlocks
                          << ", Threads = " << kernel.frameSpec.m_threadSpec.m_numThreads << std::endl;
                std::cout << "[DEBUG] Enqueueing best config kernel..." << std::endl;
#    endif
                auto bundle = recreate(kernelBundle, activeRun.userDefTuneables);
                onHost::enqueue(queue, exec, kernelptr->frameSpec, bundle);
                onHost::wait(queue);
#    ifdef DEBUG
                std::cout << "[DEBUG] Kernel execution completed for best config." << std::endl;
#    endif
                return;
            }
            else
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] Re-runs threshold not yet reached, continuing tuning." << std::endl;
#    endif
#    ifdef DEBUG
                std::cout << "[DEBUG] Calling internal_enqueue for activeRun." << std::endl;
#    endif
                internal_enqueue(
                    queue,
                    exec,
                    kernelBundle,
                    activeRun,
                    historyKernelData,
                    kernelptr->frameSpec,
                    kernelptr->sharedParams);
            }
            std::cout << " to hash: " << activeRun.toHash() << std::endl;
            std::cout << "[DEBUG] Active run maxRuns: " << activeRun.maxRuns
                      << ", History size: " << historyKernelData.runs.size()
                      << ", Sum of runs: " << historyKernelData.sumOfRuns << std::endl;
        }

        //---internal_enqueue---
        template<typename T_KernelBundle, typename T_kernelRun, typename T_NumBlocks, typename T_NumThreads>
        void internal_enqueue(
            auto const& queue,
            auto exec,
            T_KernelBundle const& kernelBundle,
            T_kernelRun& run,
            KernelData& data,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
            auto& sharedParameters)
        {
#    ifdef DEBUG
            std::cout << "[DEBUG] Internal Enqueue started." << std::endl;
#    endif
            auto runHash = run.toHash();
#    ifdef DEBUG
            std::cout << "[DEBUG] Run hash: " << runHash << std::endl;
#    endif

            if(data.runs.contains(runHash))
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] Existing run found in history. Number of runs: " << data.runs[runHash].nr_runs
                          << std::endl;
#    endif
                StorageKernelRun& storeKernel = data.runs[runHash];
                if(storeKernel.nr_runs >= this->reRuns)
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Stored run reached reRuns limit, applying strategy." << std::endl;
#    endif
                    strategy(sharedParameters, run, data.runs);
                    runHash = run.toHash(); // rehash if parameters changed
#    ifdef DEBUG
                    std::cout << "[DEBUG] Run hash updated after strategy: " << runHash << std::endl;
#    endif
                }
                else
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Stored run has room for more runs, skipping strategy." << std::endl;
#    endif
                }
            }
            else
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] No existing run found. Proceeding with new parameters." << std::endl;
                std::cout << "[DEBUG] Run hash: " << runHash << std::endl;
                for(auto const& run : data.runs)
                {
                    std::cout << "[DEBUG] Run from history: " << run.second.toHash() << std::endl;
                }
#    endif
            }

            applyCustomThreadSpec(run, spec);
#    ifdef DEBUG
            std::cout << "[DEBUG] Applied thread spec: Blocks = " << spec.m_threadSpec.m_numBlocks
                      << ", Threads = " << spec.m_threadSpec.m_numThreads << std::endl;
#    endif

            auto bundle = recreate(kernelBundle, run.userDefTuneables);
#    ifdef DEBUG
            std::cout << "[DEBUG] Kernel bundle recreated with tuneables." << std::endl;
#    endif
            {
                auto event = tune::createTimeEventFromActive(run);
#    ifdef DEBUG
                std::cout << "[DEBUG] Enqueueing kernel execution..." << std::endl;
#    endif
                onHost::enqueue(queue, exec, spec, bundle);
                onHost::wait(queue);
#    ifdef DEBUG
                std::cout << "[DEBUG] Kernel execution completed. Metric: " << run.metric << std::endl;
#    endif
            }
            ++data.sumOfRuns;
#    ifdef DEBUG
            std::cout << "[DEBUG] Kernel execution time: " << run.metric << " {" << spec.m_threadSpec.m_numBlocks
                      << "," << spec.m_threadSpec.m_numThreads << "}" << std::endl;
#    endif

            if(!data.runs.contains(runHash))
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] Adding new run to history." << std::endl;
#    endif

                data.runs[runHash] = toStore(run);
                StorageKernelRun& storeKernel = data.runs[runHash];
                using T_state = ALPAKA_TYPEOF(storeKernel.state);
                storeKernel.state = T_state::WarmUp;
                --data.sumOfRuns; // the first run of every config doesnt count towards the tuning objective
                storeKernel.nr_runs = 0;
            }
            else
            {
                StorageKernelRun& storeKernel = data.runs[runHash];
                if(storeKernel.nr_runs <= this->reRuns)
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Updating stored run. Previous best metric: " << storeKernel.metric.top()
                              << ", Previous nr_runs: " << storeKernel.nr_runs << std::endl;
#    endif
                    using T_state = ALPAKA_TYPEOF(storeKernel.state);
                    switch(storeKernel.state)
                    {
                    case T_state::WarmUp:
                        storeKernel.metric.pop(); // pop one or more initial runs
                        storeKernel.metric.push(run.metric); // we keep the same number of runs in this instance
                        storeKernel.state = T_state::Initialized;
                        ++storeKernel.nr_runs;
                        break;

                    case T_state::Initialized:
                        storeKernel.metric.push(run.metric);
                        ++storeKernel.nr_runs;
                        break;
                    default:;
                    }

#    ifdef DEBUG
                    std::cout << "[DEBUG] Updated metric: " << storeKernel.metric.top()
                              << ", nr_runs: " << storeKernel.nr_runs << ", sumOfRuns: " << data.sumOfRuns
                              << std::endl;
#    endif
                }
                else
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Stored run reached reRuns limit, skipping update." << std::endl;
#    endif
                }
            }
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
