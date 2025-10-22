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
        if(!stored.fullFlag)
            return;
        best = compareGetBest<T_MetricInterface>(best, stored);
#ifdef Debug
        std::cout << "[Best Config before]" << "," << before.toString() << "," << before.getMedian() << std::endl;
        std::cout << "[NewConfig]" << "," << stored.toString() << "," << stored.getMedian() << std::endl;
        std::cout << "[Best Config]" << "," << best.toString() << "," << best.getMedian() << std::endl;
#endif
    }

    template<typename T_MetricInterface, typename T_Config, typename T_Descriptor>
    inline void updateMetrics(
        config::ConfigRecord<T_Config>& stored,
        IO::KernelTuningMetadata<T_Config, T_Descriptor>& data,
        auto& state,
        double metric)
    {
#ifdef Debug
        std::cout << "[updateMetrics] Called with metric: " << metric << "\n";
        std::cout << "  Current state: " << static_cast<int>(stored.state) << "\n";
        std::cout << " entering with config: " << stored.toString() << std::endl;
#endif

        switch(stored.state)
        {
        case config::ConfigState::Uninitialized:
            stored.stamp = data.highestStamp + state.stamp++;
#ifdef Debug
            std::cout << "  -> State is Uninitialized. Assigned stamp: " << stored.stamp << "\n";
#endif
            break;

        case config::ConfigState::Invalid:
#ifdef Debug
            std::cout << "  -> State is Invalid. Skipping update.\n";
#endif
            return;
        case config::ConfigState::Initialized:
        case config::ConfigState::WarmUp:
        default:
#ifdef Debug
            std::cout << "  -> State is Initialized or evaluated. Proceeding.\n";
#endif
            break;
        }

        bool flagPre = stored.fullFlag;

        stored.pushMetric(metric);

        bool flagPost = stored.fullFlag;

#ifdef Debug
        std::cout << "  Pushed metric. fullFlag before: " << flagPre << ", after: " << flagPost << "\n";
        std::cout << "  nr_runs now: " << stored.nr_runs << "\n";
#endif

        bool CIcriteriaReached = (flagPre != flagPost);
        bool customCriteriaReached = (stored.nr_runs >= alpaka::tune::getRunsPerConfig());

#ifdef Debug
        std::cout << "  CI criteria reached: " << CIcriteriaReached << "\n";
        std::cout << "  Custom criteria reached: " << customCriteriaReached
                  << ", nr runs needed: " << alpaka::tune::getRunsPerConfig() << "\n";
        std::cout << "  hasRunsPerConfig_Env: " << alpaka::tune::hasRunsPerConfig_Env() << "\n";
#endif

        // update stopping criteria
        if(!alpaka::tune::hasRunsPerConfig_Env())
        {
            if(CIcriteriaReached)
            {
                ++state.numberOfCheckedConfigs;
                ++state.numValidConfigs;

#ifdef Debug
                std::cout << "  -> CI criteria met. Updated checked/valid config counts.\n";
                std::cout << "  -> Trying to assign best config.\n";
#endif
                if(!state.bestConfig.has_value())
                {
                    state.bestConfig = std::ref(stored);
                }
                else
                {
                    alpaka::tune::core::peripherals::assignBestIfBetter<T_MetricInterface>(
                        state.getBestConfig(),
                        stored);
                }
            }
        }
        else
        {
            if(CIcriteriaReached)
            {
                stored.fullFlag = false;
#ifdef Debug
                std::cout << "  -> CI criteria met, but hasRunsPerConfig enabled. Reset fullFlag = false.\n";
#endif
            }

            if(customCriteriaReached)
            {
                ++state.numberOfCheckedConfigs;
                ++state.numValidConfigs;
                stored.fullFlag = true;
#ifdef Debug
                std::cout << "  -> Custom criteria met. Marking config as full.\n";
                std::cout << "  -> Updated checked/valid config counts.\n";
                std::cout << "  -> Trying to assign best config.\n";
#endif
                if(!state.bestConfig.has_value())
                {
                    state.bestConfig = std::ref(stored);
                }
                else
                {
                    alpaka::tune::core::peripherals::assignBestIfBetter<T_MetricInterface>(
                        state.getBestConfig(),
                        stored);
                }
            }
        }
    }
} // namespace alpaka::tune::core::peripherals
#endif // UPDATEMETRIC_H
