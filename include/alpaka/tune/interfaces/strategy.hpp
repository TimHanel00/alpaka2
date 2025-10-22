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
    template<alpaka::tune::concepts::Integral T>
    inline auto vectorFromConfig(std::vector<T>& vec, alpaka::tune::concepts::ConfigLike auto const& config)
    {
        vec.resize(config.size());
        for(int i = 0; i < vec.size(); i++)
        {
            vec[i] = config[i];
        }
    }

    template<alpaka::tune::concepts::Floating T>
    inline auto vectorFromNormalizedConfig(std::vector<T>& vec, alpaka::tune::concepts::ConfigLike auto const& config)
    {
        vec.resize(config.size());
        for(int i = 0; i < config.size(); i++)
        {
            config[i] = vec[i];
        }
    }

    // Copy data *from vector into config*
    template<alpaka::tune::concepts::Integral T>
    inline auto vectorToConfig(alpaka::tune::concepts::ConfigLike auto& config, std::vector<T> const& vec)
    {
        assert(config.size() == vec.size());
        for(std::size_t i = 0; i < vec.size(); ++i)
        {
            config[i] = vec[i];
        }
    }

    // Copy normalized data *from vector into config*
    template<alpaka::tune::concepts::Floating T>
    inline auto vectorToNormalizedConfig(alpaka::tune::concepts::ConfigLike auto& config, std::vector<T> const& vec)
    {
        assert(config.size() == vec.size());
        for(std::size_t i = 0; i < vec.size(); ++i)
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
        bool increment(auto const& numValues)
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
                vectorFromConfig(initialVals, ctx.history.getOrderedHistory().front().get().config);

                currentVals = initialVals;
                init = false;
                firstStepDone = false;
                vectorToConfig(config, currentVals);
                return config;
            }
            // increment
            increment(ctx.desc.getNumValuesView());

            // check if we’ve wrapped back to initial after at last step
            if(firstStepDone && currentVals == initialVals)
            {
                ctx.env.strategyFinished = true;
                vectorToConfig(config, currentVals);
                return config;
            }

            firstStepDone = true;
            vectorToConfig(config, currentVals);
            return config;
        };
    };

    struct iterativeRefinement
    {
        std::vector<double> start; // initialized to 0.0s
        std::vector<double> end; // initialized to 1.0s
        uint32_t currentSweepCount = 0;
        uint32_t totalSweepCount = 5;
        uint32_t currentStep = 0;
        uint32_t currentDim = 0;
        uint32_t numDims = 0;
        uint32_t strideSteps = 20; // make this integer
        double zoom = 5.0;
        std::vector<double> currentVals;
        std::vector<double> initialVals;
        bool init = true;

        // --- Helper: Initialize vectors and ranges ---
        template<typename T_KernelModel, typename T_Metric>
        void initialize(StrategyContext<T_KernelModel, T_Metric> const& ctx)
        {
            auto norm = ctx.desc.createNormalizedFromConfig(ctx.history.getOrderedHistory().front().get().config);

            numDims = norm.size();
            initialVals = norm;
            currentVals = initialVals;

            start.resize(numDims);
            end.resize(numDims);

            std::ranges::fill(start, 0.0);
            std::ranges::fill(end, 1.0);

            init = false;
        }

        // --- Helper: Update current value for this dimension ---
        void updateCurrentValue()
        {
            double stepFrac = static_cast<double>(currentStep) / static_cast<double>(strideSteps);
            currentVals[currentDim] = start[currentDim] + stepFrac * (end[currentDim] - start[currentDim]);
        }

        // --- Helper: Refine (zoom in) after each dimension sweep ---
        template<typename T_KernelModel, typename T_Metric>
        void refineRange(StrategyContext<T_KernelModel, T_Metric> const& ctx)
        {
            for(std::size_t i = 0; i < numDims; ++i)
            {
                double oldDistance = end[i] - start[i];
                end[i] = currentVals[i] + oldDistance / zoom;
                start[i] = currentVals[i] - oldDistance / zoom;
            }

            currentDim = 0;
            currentSweepCount++;

            if(currentSweepCount >= totalSweepCount)
                ctx.env.strategyFinished = true;
        }

        // --- Main operator ---
        template<typename T_KernelModel, typename T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            if(init)
                initialize(ctx);

            updateCurrentValue();

            // Move to next step
            currentStep++;

            // Completed one sweep for this dimension?
            if(currentStep >= strideSteps)
            {
                currentStep = 0;

                // Snap to best performing index in this dimension if best config is available
                if(ctx.env.bestConfig.has_value())
                {
                    auto& best = ctx.desc.createNormalizedFromConfig(ctx.env.bestConfig.value().get().config);
                    currentVals[currentDim] = best[currentDim];
                }
                else
                {
                    // fall back to initial Vals if not available
                    currentVals[currentDim] = initialVals[currentDim];
                }

                // Move to next dimension
                currentDim++;

                // Completed all dimensions?
                if(currentDim >= numDims)
                    refineRange(ctx);
            }
            auto vec = vectorFromNormalizedConfig(ctx.desc.getEmptyNormalizedConfig(), currentVals);
            return vec;
        }
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
