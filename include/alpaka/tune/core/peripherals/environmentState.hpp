//
// Created by tim on 13.10.25.
//

#ifndef ENVIRONMENTSTATE_H
#define ENVIRONMENTSTATE_H

template<typename T_Config>
struct environmentState
{
    bool sessionFinished{false};
    bool strategyFinished{false};
    uint32_t numberOfCheckedConfigs{0};
    uint32_t numValidConfigs{0};
    uint32_t maxValidEvaluations{UINT32_MAX};
    uint32_t maxConfigsTotal{0};
    uint32_t stamp{0};
    uint32_t strategyLimit = Tuner_MaxConsecutiveStrategyFailures;

    auto setStrategyFinished() -> void
    {
        strategyFinished = true;
    }

    bool strategyCriteriaReached(std::optional<uint32_t> currentIndex = std::nullopt)
    {
        if(currentIndex.has_value())
        {
            if(currentIndex >= strategyLimit)
            {
                strategyFinished = true;
            }
        }
        return strategyFinished;
    }

    std::optional<std::reference_wrapper<ConfigEntry<T_Config>>> bestConfig;

    auto& getBestConfig()
    {
        return bestConfig.value().get();
    }

    uint32_t getMaxEvals() const
    {
        return std::min(maxValidEvaluations, maxConfigsTotal);
    }

    bool globalBreakCriteriaFinished()
    {
        return numValidConfigs >= maxValidEvaluations || numberOfCheckedConfigs >= maxConfigsTotal;
    }

    bool localBreakCriteriaFinished(auto const& config)
    {
        return config.fullFlag;
    }
};
#endif // ENVIRONMENTSTATE_H
