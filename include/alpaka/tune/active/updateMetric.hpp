//
// Created by tim on 03.08.25.
//

#ifndef UPDATEMETRIC_H
#define UPDATEMETRIC_H
#include "../utils/environmentVars.hpp"

#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>

namespace alpaka::tune::benchmark
{
    constexpr std::array<std::string_view, 6> ar = {"Init", "Load", "Tune", "Best", "Store", "Strategy"};

    std::string_view phaseAccessor(std::optional<uint32_t> index = std::nullopt)
    {
        static uint32_t phaseIndex = 0;
        if(index.has_value())
            phaseIndex = index.value();
        return ar[phaseIndex];
    }
} // namespace alpaka::tune::benchmark

namespace alpaka::tune::detail::internal
{
    template<typename T_MetricInterface, typename T_ConfigEntry>
    void assignBestIfBetter(T_ConfigEntry& best, T_ConfigEntry& stored)
    {
        assert(!stored.getMetrics().empty());

        if(best.getMetrics().empty() && !stored.getMetrics().empty())
        {
            best = stored;
            return;
        }
        if(!stored.fullFlag)
            return;
        best = compareGetBest<T_MetricInterface>(best, stored);
    }
} // namespace alpaka::tune::detail::internal

template<typename T_MetricInterface, typename T_Config, typename T_Descriptor>
inline void updateMetrics(
    ConfigEntry<T_Config>& stored,
    KernelData<T_Config, T_Descriptor>& data,
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
    case ConfigState::Uninitialized:
        stored.stamp = data.highestStamp + state.stamp++;
#ifdef Debug
        std::cout << "  -> State is Uninitialized. Assigned stamp: " << stored.stamp << "\n";
#endif
        break;

    case ConfigState::Dummy:
#ifdef Debug
        std::cout << "  -> State is Dummy. Skipping update.\n";
#endif
        return;
    case ConfigState::Initialized:
    case ConfigState::WarmUp:
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
                alpaka::tune::detail::internal::assignBestIfBetter<T_MetricInterface>(state.getBestConfig(), stored);
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
                alpaka::tune::detail::internal::assignBestIfBetter<T_MetricInterface>(state.getBestConfig(), stored);
            }
        }
    }
}

#endif // UPDATEMETRIC_H
