//
// Created by tim on 03.08.25.
//

#ifndef UPDATEMETRIC_H
#define UPDATEMETRIC_H
#include <alpaka/tune/IO/runTimeHistory.hpp>
#include <alpaka/tune/interfaces/MetricInterface.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>

namespace alpaka::tune::core::peripherals
{
    template<typename T_MetricInterface, typename T_Config>
    void prematureConfigSkip(config::ConfigRecord<T_Config>& stored, EnvironmentState<T_Config>& state)
    {
        auto const& best = state.getBestConfig().value().get();
        if(best == stored)
            return;
        if(stored.state != config::ConfigState::Initialized)
            return;
        auto res = best.compare(stored); // kruskal wallis comparison

        switch(res)
        {
        case alpaka::tune::config::Comparison::Greater:
            {
                stored.state = config::ConfigState::Retired;
                // best is higher then stored
                auto& config = compareGetBest<T_MetricInterface>(best, stored);
                // this returns the config that is best according to the metric interface
                // two scenarios:
                // 1. stored gets returned even though its less then best --> meaning lower is better -->
                // stored stays
                // 2. best gets returned confirming that it is better AND also significantly greater -->
                // meaning higher is better -- stored gets dropped prematurely
                if(best == config)
                {
                    stored.state = config::ConfigState::Retired;
                }
            }
        case config::Comparison::Less:
            {
                // we know best is less then stored
                auto& config = compareGetBest<T_MetricInterface>(best, stored);
                // this returns the config that is best according to the metric interface
                // two scenarios:
                // 1. stored gets returned even though its greater then best --> meaning higher is better -->
                // stored stays
                // 2. best gets returned confirming that it is better AND also significantly less -->
                // meaning lower is better -- stored gets dropped prematurely
                if(best == config)
                {
                    stored.state = config::ConfigState::Retired;
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

    template<typename T_MetricInterface, typename T_ConfigEntry>
    void assignBestIfBetter(T_ConfigEntry& best, T_ConfigEntry& stored)
    {
        assert(!stored.getMetrics().empty());
        T_ConfigEntry& before = best;
        if(best.getMeasurements().empty() && !stored.getMeasurements().empty())
        {
            best = stored;
            return;
        }
        if(!stored.state == config::ConfigState::Retired)
            return;
        best = compareGetBest<T_MetricInterface>(best, stored);
    }

    template<bool kruskalWallisSkip, typename T_MetricInterface, typename T_Config>
    inline void updateMetrics(
        config::ConfigRecord<T_Config>& stored,
        EnvironmentState<T_Config>& state,
        double_t const& metric)
    {
#ifdef Debug
        std::cout << "[updateMetrics] Called with metric: " << metric << "\n";
        std::cout << "  Current state: " << static_cast<int>(stored.state) << "\n";
        std::cout << " entering with config: " << stored.toString() << std::endl;
#endif


        bool retired = (stored.state == config::ConfigState::Retired);


        stored.pushMetric(metric);
        if(retired)
            return;
        /// update best config.
        state.template updateBestConfig<T_MetricInterface>(stored);

        bool hasCustomLocalBreakCriteria = alpaka::tune::hasRunsPerConfig_Env();
        if(!hasCustomLocalBreakCriteria)
        {
            if(stored.state == config::ConfigState::CICriteriaReached)
            {
                stored.state = config::ConfigState::Retired;
            }
            return;
        }
        bool const customCriteriaReached = (stored.nr_runs >= alpaka::tune::getRunsPerConfig());
        if(customCriteriaReached)
        {
            stored.state = config::ConfigState::Retired;
            return;
        }

        if constexpr(kruskalWallisSkip)
        {
            if(state.getBestConfig().has_value())
                prematureConfigSkip<T_MetricInterface>(stored);
        }
    }
} // namespace alpaka::tune::core::peripherals
#endif // UPDATEMETRIC_H
