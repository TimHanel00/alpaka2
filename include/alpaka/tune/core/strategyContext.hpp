//
// Created by tim on 20.10.25.
//

#ifndef STRATEGYCONTEXT_HPP
#define STRATEGYCONTEXT_HPP
#include "alpaka/tune/IO/runTimeHistory.hpp"
#include "alpaka/tune/tunable/kernelTuningModel.hpp"
#include "peripherals/environmentState.hpp"

namespace alpaka::tune
{
    template<class T_KernelModel, class T_MetricInterface>
    struct StrategyContext
    {
        using KernelModel = T_KernelModel;
        using Metric = T_MetricInterface;

        using Descriptor = alpaka::tune::ConfigDescriptor<T_KernelModel>;
        using Config = typename Descriptor::getEmptyConfig();
        using NormalizedConfig = typename Descriptor::getEmptyNormalizedConfig();
        using History = alpaka::tune::IO::ActiveHistory<Config>;
        using Environment = alpaka::tune::core::peripherals::EnvironmentState<Config>;
        using Record = alpaka::tune::config::ConfigRecord<Config>;

        // bound references (no ownership, no copies)
        Descriptor const& desc;
        History const& history;
        Environment const& env;

        // Select the better of two records based only on median and policy.
        Record& compareGetBest(Record& a, Record& b) const noexcept
        {
            if constexpr(Metric::returnComparison == detail::returnComparison::LowerIsBetter)
                return (a.getMedian() <= b.getMedian()) ? a : b;
            else
                return (a.getMedian() >= b.getMedian()) ? a : b;
        }

        // Select the worse of two records based only on median and policy.
        Record& compareGetWorst(Record& a, Record& b) const noexcept
        {
            if constexpr(Metric::returnComparison == detail::returnComparison::LowerIsBetter)
                return (a.getMedian() >= b.getMedian()) ? a : b;
            else
                return (a.getMedian() <= b.getMedian()) ? a : b;
        }
    };
} // namespace alpaka::tune
#endif // STRATEGYCONTEXT_HPP
