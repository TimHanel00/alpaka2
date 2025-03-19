//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include "alpaka/tune/IO/storageTypes.hpp"
#include "alpaka/tune/active/tuningSession.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <random>
#include <vector>

namespace alpaka::tune::strategy
{


    template<typename T_Begin, typename T_End, typename T_Stride>
    T_Begin randomIdx(IdxRange<T_Begin, T_End, T_Stride> const& range)
    {
        // Alias the vector type for the result.
        using VecType = T_Begin;

        // Create a result vector.
        VecType result;
        // Assume that T_Begin has a static member T_dim (or use T_End::T_dim).
        constexpr auto dim = IdxRange<T_Begin, T_End, T_Stride>::dim();

        // Set up a random number generator.
        // (Using static so that the generator is not re-seeded on every call.)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        // For each dimension, retrieve the minimum, maximum and stride.
        auto minVal = range.m_begin[0];
        auto maxVal = range.m_end[0];
        auto step = range.m_stride[0];

        auto numSteps = (maxVal - minVal) / step;

        // If there are no steps (or only one valid value), use minVal.
        if(numSteps <= 0)
        {
            result[0] = minVal;
        }
        else
        {
            // Choose a random step index between 0 and numSteps - 1.
            std::uniform_int_distribution<decltype(minVal)> dis(0, numSteps);
            auto k = dis(gen);
            // Set the i-th component as minVal + k * step.
            result[0] = minVal + k * step;
        }
        return result;
    }

    template<typename T, typename T_Begin, typename T_End, typename T_Stride>
    T getNextUpper(T const& value, IdxRange<T_Begin, T_End, T_Stride> const& range, bool& valid)
    {
        using VecType = T_Begin;
        // Create a result vector.
        VecType result{value};
        auto minVal = range.m_begin[0];
        auto maxVal = range.m_end[0];
        auto step = range.m_stride[0];
        result[0] = (result[0] + step);
        if(result[0] < minVal || result[0] > maxVal)
        {
            valid = false;
            return maxVal;
        }

        return T(result.product());
    }

    template<typename T, typename T_Begin, typename T_End, typename T_Stride>
    T getNextLower(T const& value, IdxRange<T_Begin, T_End, T_Stride> const& range, bool& valid)
    {
        using VecType = T_Begin;
        // Create a result vector.
        VecType result{value};
        auto minVal = range.m_begin[0];
        auto maxVal = range.m_end[0];
        auto step = range.m_stride[0];
        result[0] = (result[0] - step);
        if(result[0] < minVal || result[0] > maxVal)
        {
            valid = false;
            return minVal;
        }

        return T(result.product());
    }

    struct randomSample
    {
        template<typename T_tuneables, typename T_ActiveKernel, typename storageKernel>
        auto operator()(
            T_tuneables&& tuneables,
            [[maybe_unused]] T_ActiveKernel& kernelRun,
            [[maybe_unused]] std::unordered_map<std::string, storageKernel>& history) const
        {
            for_each(tuneables, [](auto& parameter) { parameter.value = randomIdx(parameter.idxRange).x(); });
        }
    };

    struct exhaustiveSearch
    {
        template<typename T_tuneables, typename T_ActiveKernel, typename storageKernel>
        auto operator()(
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, storageKernel>& history) const
        {
            if(history.contains(kernelRun.toHash()))
            {
                for_each(
                    tuneables,
                    [&history, &kernelRun](auto& parameter)
                    {
                        constexpr auto dim = static_cast<std::size_t>(1);
                        //@TODO make this dynamic but ALPAKA_TYPE_OF(parameter->idxRange)::dim() did no get deduced
                        // correctly on GPU
                        using type = std::size_t;
                        auto initialValue = parameter.value;
                        bool valid = true;
                        while(valid)
                        {
                            parameter.value = getNextUpper(parameter.value, parameter.idxRange, valid);
                            if(!history.contains(kernelRun.toHash()))
                                return;
                            if(history[kernelRun.toHash()].nr_runs < getReRuns())
                                return;
                        }
                        valid = true;
                        while(valid)
                        {
                            parameter.value = getNextLower(parameter.value, parameter.idxRange, valid);
                            if(!history.contains(kernelRun.toHash()))
                                return;
                            if(history[kernelRun.toHash()].nr_runs < getReRuns())
                                return;
                        }

                        parameter.value = initialValue;
                    });
            }
        }
    };

    struct randomSearch
    {
        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, StorageKernelRun>& history) const
        {
            randomSample{}(tuneables, kernelRun, history);
            if(history.contains(kernelRun.toHash()))
            {
                exhaustiveSearch{}(tuneables, kernelRun, history);
            }
        };
    };

    //@TODO move to different namespace
    struct bestRecorded
    {
        template<typename T_ActiveKernel>
        auto operator()(T_ActiveKernel& kernelRun, std::unordered_map<std::string, StorageKernelRun>& history)
        {
            auto best = history.begin()->second;
            for(auto& run : history)
            {
                if(run.second.metric < best.metric)
                {
                    best = run.second; // Update selectedRun to the run with the smaller time
                }
            }
            toActive(kernelRun, best);
        }
    };

    struct initialValues
    {
        template<typename tuneables, typename T_KernelRun, typename KernelRun>
        auto operator()(
            std::vector<std::shared_ptr<tuneables>>& tuningParameters,
            T_KernelRun& kernelRun,
            std::unordered_map<std::string, KernelRun>& history) const
        {
            return tuningParameters;
        }
    };
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
