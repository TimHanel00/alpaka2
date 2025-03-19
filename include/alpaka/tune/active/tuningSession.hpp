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
            std::cout << " CONFIG: " << config << std::endl;
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
            ret.config = this->config;
            std::cout << " CONFIG: " << ret.config << std::endl;
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
            std::cout << " CONFIG: " << ret.config << std::endl;
            return ret;
        }

        auto withGridSizeTune()
        {
            TuningSession<T_Strategy, tune::GridSizeTune<>, T_BlockSize, true, block> ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune{});
            ret.config = this->config;
            std::cout << " CONFIG: " << ret.config << std::endl;
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
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withBlockSizeTune(tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSession<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride>, grid, true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = tune;
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
                if(config != "")
                {
                    history.loadConfig(config);
                }
            }
            static auto& kernel = createKernelSingleton<grid, block>(
                device,
                exec,
                frameSpec,
                kernelBundle,
                this->run,
                sessionSpecifier,
                history);
            auto& activeRun = *kernel.activeRunPtr;
            auto& ptrToHistory = kernel.ptrToHistory;
            if(ptrToHistory->runs.size() >= activeRun.maxRuns)
            {
                // if the tuning space is exhausted, always take the best tune
                {
                    auto event = tune::createTimeEventFromActive(activeRun);
                    alpaka::tune::strategy::bestRecorded{}(activeRun, ptrToHistory->runs);
                    applyCustomThreadSpec(activeRun, kernel.frameSpec);
                    auto bundle = recreate(kernelBundle, activeRun.tuneables);
                    onHost::enqueue(queue, exec, kernel.frameSpec, bundle);
                    onHost::wait(queue);
                }
            }
            else
            {
                internal_enqueue(queue, exec, kernelBundle, activeRun, ptrToHistory.get(), kernel.frameSpec);
            }
        }

        template<typename T_KernelBundle, typename T_kernelRun, typename T_NumBlocks, typename T_NumThreads>
        void internal_enqueue(
            auto const& queue,
            auto exec,
            T_KernelBundle const& kernelBundle,
            T_kernelRun& run,
            KernelData* data,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec)
        {
            auto runHash = run.toHash();
            if(data->runs.contains(runHash)) // check whether this parameter tuple was already taken
            {
                StorageKernelRun& storeKernel = data->runs[runHash];
                if(storeKernel.nr_runs >= this->reRuns) // check whether stored run has less runs
                {
                    auto sharedParams = makeSharedParameterInterface<grid, block, T_kernelRun>(run);
                    strategy(sharedParams, run, data->runs);
                    runHash = run.toHash();
                }
            }
            applyCustomThreadSpec(run, spec);

            {
                auto bundle = recreate(kernelBundle, run.tuneables);

                auto event = tune::createTimeEventFromActive(run);

                onHost::enqueue(queue, exec, spec, bundle);
                onHost::wait(queue);
            }
            if(!data->runs.contains(runHash))
            {
                data->runs[runHash] = toStore(run);
            }
            else
            {
                StorageKernelRun& storeKernel = data->runs[runHash];

                if(storeKernel.nr_runs <= this->reRuns)
                {
                    storeKernel.metric
                        = ((this->reRuns - storeKernel.nr_runs) * storeKernel.metric + run.metric) / this->reRuns;
                    ++storeKernel.nr_runs;
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
