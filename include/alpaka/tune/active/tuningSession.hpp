//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE

#ifdef ENABLE_AUTOTUNE
#    include "alpaka/onHost.hpp"
#    include "alpaka/tune/active/KernelSingleton.hpp"
#    include "alpaka/tune/utils/environmentVars.hpp"

#    include <alpaka/tune/IO/tuningHistory.hpp>
#    include <alpaka/tune/active/strategy.hpp>
#    include <alpaka/tune/utils/TimeEvent.hpp>
#    include <alpaka/tune/utils/tupleHandle.hpp>

#    include <cmath>
#    include <iostream>
#    include <numeric>
#    include <string>
#    include <unordered_map>
#    include <variant>

namespace alpaka
{
    template<typename T_KernelRun, typename T_FrameSpec>
    static T_FrameSpec& applyCustomThreadSpec(T_KernelRun& kernelRun, T_FrameSpec& spec)
    {
        if(kernelRun.gridSize != std::nullopt)
        {
            auto val = kernelRun.gridSize->value;
            using tuneableGridType = ALPAKA_TYPEOF(val);
            if constexpr(alpaka::isVector_v<tuneableGridType>)
            {
                if constexpr(std::is_same_v<tuneableGridType, ALPAKA_TYPEOF(spec.m_threadSpec.m_numBlocks)>)
                {
                    spec.m_threadSpec.m_numBlocks = val;
                }
                else
                {
                    throw std::runtime_error(
                        "TuningSession::applyCustomThreadSpec(): invalid custom gridSize tuning - must conform with "
                        "type of numFrames");
                }
            }
            else
            {
                spec.m_threadSpec.m_numBlocks = ALPAKA_TYPEOF(spec.m_threadSpec.m_numBlocks)(val);
            }
        }
        if(kernelRun.threadBlockSize != std::nullopt)
        {
            auto val = kernelRun.threadBlockSize->value;
            using tuneableBLockType = ALPAKA_TYPEOF(val);
            if constexpr(alpaka::isVector_v<tuneableBLockType>)
            {
                if constexpr(std::is_same_v<tuneableBLockType, ALPAKA_TYPEOF(spec.m_threadSpec.m_numThreads)>)
                {
                    spec.m_threadSpec.m_numThreads = val;
                }
                else
                {
                    throw std::runtime_error(
                        "TuningSession::applyCustomThreadSpec(): invalid custom threadBlockSize tuning - must conform "
                        "with type of frameExtent");
                }
            }
            else
            {
                spec.m_threadSpec.m_numThreads = ALPAKA_TYPEOF(spec.m_threadSpec.m_numThreads)(val);
            }
        }
        return spec;
    }

    template<typename T_active>
    void printGBFromActive(T_active& active)
    {
        if(active.gridSize.has_value())
        {
            std::cout << "G: " << active.gridSize->valueToString() << std::endl;
            return;
        }
        if(active.threadBlockSize.has_value())
        {
            std::cout << "B: " << active.threadBlockSize->valueToString() << std::endl;
            return;
        }

        if(active.threadBlockSize.has_value() && active.gridSize.has_value())
        {
            std::cout << "G: " << active.gridSize->valueToString()
                      << " , B: " << active.threadBlockSize->valueToString() << std::endl;
        }
    }

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()

    template<
        typename T_Strategy = alpaka::tune::strategy::randomSearch,
        typename T_GridSize = alpaka::tune::GridSizeTune<>,
        typename T_BlockSize = alpaka::tune::ThreadBlockSizeTune<>,
        bool grid = false,
        bool block = false>
    struct TuningSession
    {
        using T_floating = double_t;
        using T_Integer = std::size_t;
        ActiveKernelRun<T_GridSize, T_BlockSize, T_floating> run;
        tune::TuningHistory history;
        T_Strategy strategy;
        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;

        explicit TuningSession(T_Strategy strategy) : strategy(strategy), reRuns(getReRuns())
        {
            run.metric = MetricUndefined;
        }

        /*
        TuningSession(const TuningSession&) = delete;
        TuningSession& operator=(const TuningSession&) = delete;
        TuningSession(TuningSession&&) noexcept = default;
        TuningSession& operator=(TuningSession&&) noexcept = default;*/
        TuningSession& withReRuns(std::size_t const& reRuns)
        {
            if(getReRuns() != 0)
            {
                this->reRuns = reRuns;
            }
            return *this;
        }

        TuningSession& withConfig(std::string const& config)
        {
            this->config = config;
            return *this;
        }

        template<typename... T_Specifiers>
        TuningSession& withRunSpecifiers(T_Specifiers... specifiers)
        {
            processArgs(sessionSpecifier, specifiers...);

            return *this;
        }

        template<bool copyGridSize = false, bool copyBlockSize = false, typename T_newSession>
        void copy(T_newSession& session)
        {
            session.config = this->config;
            session.dynamicRuns_Nr = dynamicRuns_Nr;
            session.sessionSpecifier = sessionSpecifier;
            session.run.metric = this->run.metric;
            if constexpr(copyGridSize)
            {
                session.run.gridSize = this->run.gridSize;
            }
            else if constexpr(copyBlockSize)
            {
                session.run.threadBlockSize = this->run.threadBlockSize;
            }
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withGridSizeTune(tune::GridSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSession<T_Strategy, tune::GridSizeTune<T, T_Begin, T_End, T_Stride>, T_BlockSize, true, block> ret{
                strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = tune;
            ret.run.gridSize->userDef = true;
            ret.config = this->config;
            return ret;
        }

        template<typename T, auto dim>
        auto withGridSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSession<T_Strategy, tune::GridSizeTune<VecType, VecType, VecType, VecType>, T_BlockSize, true, block>
                ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune<VecType, VecType, VecType, VecType>{tune});
            ret.config = this->config;
            ret.run.gridSize->userDef = true;
            return ret;
        }

        auto withGridSizeTune()
        {
            TuningSession<T_Strategy, tune::GridSizeTune<>, T_BlockSize, true, block> ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune{});
            ret.config = this->config;
            return ret;
        }

        auto withBlockSizeTune()
        {
            TuningSession<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<>, grid, true> ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune{});
            ret.config = this->config;
            return ret;
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSession<
                T_Strategy,
                T_GridSize,
                tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>,
                grid,
                true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>{tune});
            ret.config = this->config;
            ret.run.threadBlockSize->userDef = true;
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withBlockSizeTune(tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSession<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride>, grid, true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune);
            ret.run.threadBlockSize->userDef = true;
            ret.config = this->config;
            return ret;
        }

        auto& withDynamicRuns(std::size_t runs)
        {
            dynamicRuns_Nr = static_cast<T_Integer>(runs);
            return *this;
        }

        /** Enqueue and Execute a kernel for the tuning session will run @DynamicRuns times
         * @param device
         * @param queue the kernel will be executed after all previous work in this queue is finished
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
            if(!m_initialized)
            {
                m_initialized = true;
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
            static auto& kernel = createKernelSingleton<grid, block>(
                device,
                exec,
                frameSpec,
                kernelBundle,
                this->run,
                sessionSpecifier,
                history);
#    ifdef DEBUG
            std::cout << "[DEBUG] Kernel singleton created." << std::endl;
#    endif

            auto& activeRun = *kernel.activeRunPtr;
            auto& ptrToHistory = kernel.ptrToHistory;
            std::cout << "[DEBUG] Active run maxRuns: " << activeRun.maxRuns
                      << ", History size: " << ptrToHistory->runs.size()
                      << ", Sum of runs: " << ptrToHistory->sumOfRuns << std::endl;

            if(ptrToHistory->sumOfRuns >= getMaxRuns() || ptrToHistory->sumOfRuns >= activeRun.maxRuns * getReRuns())
            {
                std::cout << "[DEBUG] Tuning space exhausted, selecting best config." << std::endl;
                auto event = tune::createTimeEventFromActive(activeRun);
                alpaka::tune::strategy::bestRecorded{}(activeRun, ptrToHistory->runs);
                applyCustomThreadSpec(activeRun, kernel.frameSpec);
#    ifdef DEBUG
                std::cout << "[DEBUG] Best config applied: Blocks = " << kernel.frameSpec.m_threadSpec.m_numBlocks
                          << ", Threads = " << kernel.frameSpec.m_threadSpec.m_numThreads << std::endl;
                std::cout << "[DEBUG] Enqueueing best config kernel..." << std::endl;
#    endif
                auto bundle = recreate(kernelBundle, activeRun.tuneables);
                onHost::enqueue(queue, exec, kernel.frameSpec, bundle);
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
                    ptrToHistory.get(),
                    kernel.frameSpec,
                    kernel.sharedParams);
            }
        }

        // --- internal_enqueue ---

        template<typename T_KernelBundle, typename T_kernelRun, typename T_NumBlocks, typename T_NumThreads>
        void internal_enqueue(
            auto const& queue,
            auto exec,
            T_KernelBundle const& kernelBundle,
            T_kernelRun& run,
            KernelData* data,
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

            if(data->runs.contains(runHash))
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] Existing run found in history. Number of runs: " << data->runs[runHash].nr_runs
                          << std::endl;
#    endif
                StorageKernelRun& storeKernel = data->runs[runHash];
                if(storeKernel.nr_runs >= this->reRuns)
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Stored run reached reRuns limit, applying strategy." << std::endl;
#    endif
                    strategy(sharedParameters, run, data->runs);
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
                for(auto const& run : data->runs)
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

            auto bundle = recreate(kernelBundle, run.tuneables);
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

#    ifdef DEBUG
            std::cout << "[DEBUG] Kernel execution time: " << run.metric << " {" << spec.m_threadSpec.m_numBlocks
                      << "," << spec.m_threadSpec.m_numThreads << "}" << std::endl;
#    endif

            if(!data->runs.contains(runHash))
            {
#    ifdef DEBUG
                std::cout << "[DEBUG] Adding new run to history." << std::endl;
#    endif
                data->runs[runHash] = toStore(run);
            }
            else
            {
                StorageKernelRun& storeKernel = data->runs[runHash];
                if(storeKernel.nr_runs <= this->reRuns)
                {
#    ifdef DEBUG
                    std::cout << "[DEBUG] Updating stored run. Previous metric: " << storeKernel.metric
                              << ", Previous nr_runs: " << storeKernel.nr_runs << std::endl;
#    endif
                    storeKernel.metric
                        = (storeKernel.metric * storeKernel.nr_runs + run.metric) / (storeKernel.nr_runs + 1);
                    ++storeKernel.nr_runs;
                    ++data->sumOfRuns;
#    ifdef DEBUG
                    std::cout << "[DEBUG] Updated metric: " << storeKernel.metric
                              << ", nr_runs: " << storeKernel.nr_runs << ", sumOfRuns: " << data->sumOfRuns
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
            if(m_initialized && config != "")
            {
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
