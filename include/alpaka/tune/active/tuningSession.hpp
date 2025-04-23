//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE
#include "../../../../example/heatEquation2D/src/StencilKernel.hpp"

#ifdef ENABLE_AUTOTUNE

#    include <alpaka/tune/active/sessionBuilder.h>

namespace alpaka::tune::detail::internal
{
    template<
        typename T_Device,
        typename T_Exec,
        typename T_NumFrames,
        typename T_FrameExtent,
        typename T_KernelBundle,
        typename T_Run,
        typename T_SessionSpecifier,
        typename T_History>
    auto setup_enqueue(

        T_Device device,
        T_Exec exec,
        alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
        T_KernelBundle const& kernelBundle,
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

        static auto* kernelptr
            = getTuningEnvironment(device, exec, frameSpec, kernelBundle, run, sessionSpecifier, history).get();

        if(sessionSpecifier != kernelptr->ptrToHistory->specifiers)
        {
            kernelptr
                = getTuningEnvironment(device, exec, frameSpec, kernelBundle, run, sessionSpecifier, history).get();
        }

        return kernelptr;
    }

    // Strategy functor and Constraint functor must be passed externally now

    // Check if a strategy should be applied and config should be skipped
    template<typename Run, typename Data, typename SharedParams, typename Strategy>
    bool shouldSkipDueToHistory(Run& run, Data& data, SharedParams& sharedParams, Strategy& strategy)
    {
        auto runHash = run.toHash();

        if(!data.runs.contains(runHash))
        {
            return false;
        }

        auto& stored = data.runs[runHash];

        if(stored.nr_runs >= getRunsPerConfig() && stored.fullFlag)
        {
            if(data.nrOfConfigs == getMaxRuns(run.maxRuns) - 1)
            {
                ++data.nrOfConfigs;
                return true;
            }

            std::string oldHash = run.toHash();

            strategy(sharedParams, run, data);

            std::string newHash = run.toHash();

            if(newHash != oldHash && !data.runs.contains(newHash))
            {
                ++data.nrOfConfigs;
            }

            return true;
        }
        return false;
    }

    // Validate constraint or mark as Dummy
    template<typename Constraint, typename Run, typename Data>
    bool violatesConstraint(Run& run, Data& data, std::string const& runHash, Constraint& constraint)
    {
        if(data.runs.contains(runHash))
        {
            auto const& stored = data.runs.at(runHash);
            return stored.state != StorageKernelRun::State::Dummy;
        }

        if(!constraint(run))
        {
            data.runs[runHash] = toStore(run);
            auto& stored = data.runs[runHash];
            using T_state = ALPAKA_TYPEOF(stored.state);
            stored.state = T_state::Dummy;
            stored.fullFlag = true;
            stored.nr_runs = 0;
            return false;
        }

        return true;
    }

    inline void applyConfigAndExecuteKernel(
        auto const& queue,
        auto exec,
        auto const& kernelBundle,
        auto& spec,
        auto& run)
    {
        applyCustomThreadSpec(run, spec);
        auto bundle = recreate(kernelBundle, run.userTuneables);
        {
            auto event = tune::createTimeEventFromActive(run);
            onHost::enqueue(queue, exec, spec, bundle);
            onHost::wait(queue);
        }
    }

    // Extracts logic when max configs is reached and best config should be applied
    // Should be called inside internal_enqueue()
    template<typename T_KernelBundle, typename T_kernelRun, typename T_NumBlocks, typename T_NumThreads>
    void applyBestAndExecute(
        auto const& queue,
        auto exec,
        T_KernelBundle const& kernelBundle,
        T_kernelRun& run,
        KernelData& data,
        onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
        auto& history,
        auto const& config)
    {
        static bool write = true;
        if(write)
        {
            history.storeConfig(config);
            write = false;
        }

        alpaka::tune::strategy::bestRecorded<alpaka::tune::Timing>{}(run, data);
        applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, run);
        onHost::wait(queue);
    }

    inline void storeOrUpdateMetrics(auto& run, KernelData& data)
    {
        auto runHash = run.toHash();
        using T_state = ALPAKA_TYPEOF(data.runs[runHash].state);

        if(!data.runs.contains(runHash))
        {
            data.runs[runHash] = toStore(run);
            auto& stored = data.runs[runHash];
            stored.stamp = run.m_strategyState.configStamp++;
            stored.state = T_state::WarmUp;
            stored.nr_runs = 0;

            if(stored.state == T_state::Dummy)
            {
                stored.fullFlag = true;
            }

            return;
        }

        auto& stored = data.runs[runHash];
        if(stored.state == T_state::Dummy || (stored.nr_runs > getRunsPerConfig() && stored.fullFlag))
        {
            return;
        }

        switch(stored.state)
        {
        case T_state::WarmUp:
            stored.metricContainer.pop();
            stored.pushMetric(run.metric);
            stored.state = T_state::Initialized;
            ++stored.nr_runs;
            break;

        case T_state::Initialized:
            stored.pushMetric(run.metric);
            ++stored.nr_runs;
            break;

        default:
            break;
        }
    }
} // namespace alpaka::tune::detail::internal

namespace alpaka
{
    template<typename T_FrameSpec, typename... T_Args>
    static T_FrameSpec& applyCustomThreadSpec(ActiveKernelRun<T_Args...>& kernelRun, T_FrameSpec& spec)
    {
        if constexpr(ActiveKernelRun<T_Args...>::hasNumFramesTune())
        {
            spec.m_numFrames = kernelRun.getNumFramesTune().value;
        }
        if constexpr(ActiveKernelRun<T_Args...>::hasFrameExtentTune())
        {
            spec.m_frameExtent = kernelRun.getFrameExtentTune().value;
        }
        if constexpr(ActiveKernelRun<T_Args...>::hasNumBlocksTune())
        {
            spec.m_threadSpec.m_numBlocks = kernelRun.getNumBlocksTune().value;
        }
        if constexpr(ActiveKernelRun<T_Args...>::hasThreadBlockSizeTune())
        {
            spec.m_threadSpec.m_numThreads = kernelRun.getThreadBlockSizeTune().value;
        }

        return spec;
    }

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()

    template<typename T_Strategy = tune::strategy::randomSearch<tune::Timing>, typename... T_KernelRunArgs>
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
                this->run,
                sessionSpecifier,
                history,
                config);

            auto& activeRun = *kernelptr->activeRunPtr;
            KernelData& historyKernelData = (*kernelptr->ptrToHistory);
            using T_Context = ALPAKA_TYPEOF(kernelptr);


            internal_enqueue<T_Context>(
                queue,
                exec,
                kernelBundle,
                activeRun,
                historyKernelData,
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
            T_kernelRun& run,
            KernelData& data,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec,
            auto& sharedParameters)
        {
            using namespace alpaka::tune::detail::internal;
            while(data.nrOfConfigs < getMaxRuns(run.maxRuns) && !run.m_strategyState.done)
            {
                if(shouldSkipDueToHistory(run, data, sharedParameters, strategy))
                    continue;
                // if(violatesConstraint(run, data, runHash, constraint))
                // continue;

                break;
            }
            if(data.nrOfConfigs >= getMaxRuns(run.maxRuns))
            {
                applyBestAndExecute(queue, exec, kernelBundle, run, data, spec, history, config);
                std::cout << "[TUNE] with best config " << run.toHash() << " time: " << run.metric << std::endl;
                return;
            }

            applyConfigAndExecuteKernel(queue, exec, kernelBundle, spec, run);
            std::cout << "[TUNE] " << data.nrOfConfigs << " out of " << getMaxRuns(run.maxRuns) << " checked. "
                      << "Current config: " << run.toHash() << " time: " << run.metric << std::endl;
            storeOrUpdateMetrics(run, data);
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
