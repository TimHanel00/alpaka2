//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include "alpaka/tune/core/strategyContext.hpp"

#include <alpaka/tune/core/peripherals/environmentState.hpp>
#include <alpaka/tune/utils/Random.hpp>

#if defined(strategy_bayesianOptimization)
#    include <alpaka/tune/interfaces/strategy_Impl/bayesionOptimizer.hpp>
#endif
#if defined(strategy_simulatedAnnealing)
#    include <alpaka/tune/interfaces/strategy_Impl/simA_oldStub.hpp>
#endif
namespace alpaka::tune::strategy
{
    template<typename T>
    inline auto vectorFromConfig(std::vector<T>& vec, alpaka::tune::concepts::Config auto const& config)
    {
        vec.resize(config.size());
        for(int i = 0; i < vec.size(); i++)
        {
            vec[i] = config[i];
        }
    }

    template<typename T>
    inline auto vectorFromNormalizedConfig(std::vector<T>& vec, alpaka::tune::concepts::Config auto const& config)
    {
        assert(vec.size() == config.size());
        for(int i = 0; i < config.size(); i++)
        {
            config[i] = vec[i];
        }
    }

    struct randomSearch
    {
    };

    struct exhaustiveSearch
    {
        std::vector<uint32_t> currentVals;
        std::vector<uint32_t> initialVals;
        std::vector<uint32_t> numValues;
        std::size_t maxPossible = 1;
        std::size_t currentIndex = 0;
        std::size_t startingIndex = 0;
        std::size_t currentCount = 0;
        bool init = true;
        bool firstStepDone = false;

        // mixed-radix counter
        bool increment()
        {
            for(std::size_t i = currentVals.size(); i-- > 0;)
            {
                currentVals[i]++;
                if(currentVals[i] < numValues[i])
                    return true;
                currentVals[i] = 0;
            }
            return false;
        }

        template<concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            auto config = ctx.desc.getEmptyConfig();
            if(init)
            {
                vectorFromConfig(initialVals, ctx.history.getInsertionOrder().front().get().config);
                vectorFromConfig(numValues, ctx.desc.getNumValues());

                currentVals = initialVals;
                init = false;
                firstStepDone = false;
                return vectorToConfig(currentVals, config);
            }
            // increment
            increment();

            // check if we’ve wrapped back to initial after at last step
            if(firstStepDone && currentVals == initialVals)
            {
                ctx.env.strategyFinished = true;
                return vectorToConfig(currentVals, config);
            }

            firstStepDone = true;
            return vectorToConfig(currentVals, config);
        };
    };

    struct iterativeRefinement
    {
        double_t start = 0.0;
        double_t end = 0.0;
        uint32_t currentSweepCount = 0;
        uint32_t currentStep = 0;
        uint32_t currentDim = 0;
        uint32_t numDims;
        double_t strideSteps = 20;
        double_t zoom = 5;
        std::vector<double_t> currentVals;
        bool init = true;

        template<concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            if(init)
            {
                auto config = typename StrategyContext<T_KernelModel, T_Metric>::NormalizedConfig{};
                vectorFromConfig(
                    currentVals,
                    ctx.desc.createNormalizedFromConfig(ctx.history.getInsertionOrder().front().get().config));
                init = false;
                numDims = config.size();
            }

            currentVals[currentDim] = currentStep++ * (1.0) / strideSteps;
            if(currentStep == strideSteps)
        };
    };

    struct randomSample
    {
        std::uniform_real_distribution<double> m_dist{0.0, 1.0};

        template<concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            auto config = typename StrategyContext<T_KernelModel, T_Metric>::NormalizedConfig{};
            for(auto& val : config)
                val = m_dist(RNG::get());
            return config;
        };
    };
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
