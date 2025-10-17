//
// Created by tim on 05.02.25.
//
#ifndef TUNINGSESSION_H
#define TUNINGSESSION_H

#include <alpaka/tune/core/peripherals/constraint.hpp>
#include <alpaka/tune/core/sessionBuilder.hpp>
#include <alpaka/tune/tuneable/frameSpecTuningModel.hpp>
#include <alpaka/tune/tuneable/kernelTuningModel.hpp>
#include <alpaka/tune/utils/compileTimeTemplates.hpp>

#include <utility>

namespace alpaka::tune::detail::internal
{
    template<
        typename T_Queue,
        typename T_Exec,
        typename T_KernelBundle,
        typename T_FrameSpecTuningModel,
        typename T_Session>
    auto* setup_enqueue(
        T_Queue& queue,
        T_Exec exec,
        T_FrameSpecTuningModel const& frameSpecTune,
        T_KernelBundle const& kernelBundle,
        T_Session const& session)
    {
        static auto* kernelptr = getTuningEnvironment(queue, exec, frameSpecTune, kernelBundle, session).get();
        auto& data = kernelptr->env_kernelData;
        // always use the same context unless session specifier change.
        if(session.sessionSpecifier != data.specifiers)
        {
            kernelptr = getTuningEnvironment(queue, exec, frameSpecTune, kernelBundle, session).get();
        }
        if(!data.histEvaluated)
        {
            data.histEvaluated = true;
            auto& environment_state = kernelptr->environmentState;
            if(environment_state.globalBreakCriteriaFinished())
            {
                environment_state.sessionFinished = true;
            }
        }
        return kernelptr;
    }

#define maxConsecutiveStrategyRuns 4000

    // Check if a m_strategy should be applied and Config should be skipped


    // Validate constraint or mark as Invalid

#define WarmUpRuns 1


} // namespace alpaka::tune::detail::internal

namespace alpaka::tune
{
    template<
        concepts::Strategy T_Strategy = tune::strategy::randomSearch,
        concepts::MetricInterface T_MetricInterface = tune::metricInterface::Timing,
        typename T_Constraints = std::tuple<>>
    struct TuningSession
    {
        T_Strategy m_strategy;
        T_MetricInterface m_metricInterface;
        T_Constraints m_constraintTuple;
        std::string config;
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;
        uint32_t finishedEnvironment = 0;

        explicit TuningSession(
            T_Strategy const& strategy,
            T_MetricInterface const& interface,
            T_Constraints const& constraints,
            std::string config,
            std::vector<std::string> const& sessionSpecifiers)
            : m_strategy(strategy)
            , m_metricInterface(interface)
            , m_constraintTuple(constraints)
            , config(std::move(config))
            , sessionSpecifier(sessionSpecifiers)
        {
        }

        /** Enqueue and Execute a kernel for the tuning session
         * @param queue
         * @param exec
         * @param frameSpecTune
         * @param frameSpec
         * @param kernelBundle the compute kernel and there arguments
         */
        template<
            typename T_Queue,
            typename T_Exec,
            typename T_FramesTune,
            typename T_FrameExtentTune,
            typename T_ThreadTune,
            typename T_BlockTune,
            typename T_KernelBundle>
        auto enqueue(
            T_Queue& queue,
            T_Exec exec,
            FrameSpecTuningModel<T_FramesTune, T_FrameExtentTune, T_ThreadTune, T_BlockTune> const& frameSpecTune,
            T_KernelBundle const& kernelBundle)
        {
            auto* environmentPtr
                = tune::detail::internal::setup_enqueue<T_MetricInterface>(queue, exec, frameSpecTune, *this);

            bool bef = environmentPtr->readyForTerminate;
            environmentPtr->launch(queue, exec, frameSpecTune, kernelBundle);
            if(bef != environmentPtr->readyForTerminate)
            {
                finishedEnvironment++;
            }
        }

        /** Enqueue and Execute a kernel for the tuning session
         * @param queue
         * @param exec
         * @param frameSpec
         * @param kernelBundle the compute kernel and there arguments
         */
        template<
            typename T_Queue,
            typename T_Exec,
            typename T_NumFrames,
            typename T_FrameExtent,
            typename T_ThreadSpec,
            typename T_KernelBundle>
        auto enqueue(
            T_Queue& queue,
            T_Exec exec,
            onHost::FrameSpec<T_NumFrames, T_FrameExtent, T_ThreadSpec>& frameSpec,
            T_KernelBundle const& kernelBundle)
        {
            return enqueue(queue, exec, FrameSpecTuningModel{frameSpec}, kernelBundle);
        }

        //@TODO implement a getBestValues method, implement a function based version (no queue, no exec, no
        // frameSpec,...).
        ~TuningSession()
        {
#ifdef Debug
            std::cout << " destructor called" << std::endl;
#endif
        }
    };

    // KernelData stores Tuneable parameters as objects

    /*
     * is holding
     */

} // namespace alpaka::tune


#endif // TUNINGSESSION_H
