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


    template<typename T>
    T randomIdx(IdxRangeHandle<T> const& range, auto& value)
    {
        // Alias the vector type for the result.

        // Create a result vector.
        T result;
        // Assume that T_Begin has a static member T_dim (or use T_End::T_dim).

        // Set up a random number generator.
        // (Using static so that the generator is not re-seeded on every call.)
        static std::random_device rd;
        static std::mt19937 gen(rd());
        // For each dimension, retrieve the minimum, maximum and stride.
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        assert(step != 0 && "Stride of Tuneable must be non-negative!");
        assert(value >= minVal && value <= maxVal && "Value of Tuneable is not in idxRange");
        using VecType = ALPAKA_TYPEOF(minVal);
        auto numStepsUp = (maxVal - VecType(value)) / abs(step);
        auto numStepsDown = (VecType(value) - minVal) / abs(step);
        std::uniform_int_distribution<decltype(minVal)> dis(0, numStepsUp + numStepsDown);
        auto k = dis(gen);
        if(k > numStepsUp)
        {
            std::uniform_int_distribution<decltype(minVal)> disD(0, numStepsDown);
            auto numStep = disD(gen);
            result = value - (numStep * step);
        }
        else
        {
            std::uniform_int_distribution<decltype(minVal)> disU(0, numStepsUp);
            auto numStep = disU(gen);
            result = value + (numStep * step);
        }
        return result;
    }

    template<typename T>
    T getNextUpper(T const& value, IdxRangeHandle<T> const& range, bool& valid)
    {
        // Create a result vector.
        T result{value};
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        result = (result + step);
        if(result < minVal || result > maxVal)
        {
            valid = false;
            return maxVal;
        }

        return result;
    }

    template<typename T>
    T getNextLower(T const& value, IdxRangeHandle<T> const& range, bool& valid)
    {
        // Create a result vector.
        T result{value};
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        result = (result - step);
        if(result < minVal || result > maxVal)
        {
            valid = false;
            return minVal;
        }

        return result;
    }

    struct randomSample
    {
        template<typename T_tuneables, typename T_ActiveKernel, typename storageKernel>
        auto operator()(
            T_tuneables&& tuneables,
            [[maybe_unused]] T_ActiveKernel& kernelRun,
            [[maybe_unused]] std::unordered_map<std::string, storageKernel>& history) const
        {
            for_each(
                tuneables,
                [](auto& parameter) { parameter.value = randomIdx(parameter.idxRange, parameter.value); });
        };
    };

    template<std::size_t I = 0, typename Func, typename Tuple>
    inline void for_each_enumerate(std::size_t idx, Tuple& tuple, Func&& f)
    {
        if constexpr(I < std::tuple_size_v<std::remove_reference_t<Tuple>>)
        {
            if(idx == I)
            {
                f(std::get<I>(tuple));
            }
            else
            {
                for_each_enumerate<I + 1>(idx, tuple, std::forward<Func>(f));
            }
        }
    }

    struct exhaustiveSearch
    {
        template<typename Tuple, typename T_ActiveKernel, typename StorageKernel>
        void recurse(
            Tuple& tuneables,
            std::size_t dim,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, StorageKernel>& history,
            bool& found) const
        {
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
            auto hash = kernelRun.toHash();
            if(found || !history.contains(hash) || history[hash].nr_runs < getReRuns())
            {
                found = true;
                return;
            }
            // new tuneable found
            if(dim == N)
            {
                return;
            }

            for_each_enumerate(
                dim,
                tuneables,
                [&](auto& param)
                {
                    auto oldValue = param.value;

                    // Try all values in the index range for this parameter
                    {
                        bool valid = true;
                        auto value = param.value;
                        while(valid && !found)
                        {
                            param.value = value;
                            recurse(tuneables, dim + 1, kernelRun, history, found);
                            value = getNextUpper(value, param.idxRange, valid);
                        }
                    }
                    {
                        bool valid = true;
                        auto value = getNextLower(oldValue, param.idxRange, valid);
                        while(valid && !found)
                        {
                            param.value = value;
                            recurse(tuneables, dim + 1, kernelRun, history, found);
                            value = getNextLower(value, param.idxRange, valid);
                        }
                    }
                    if(!found)
                        param.value = oldValue;
                });
        }

        template<typename T_tuneables, typename T_ActiveKernel, typename storageKernel>
        auto operator()(
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, storageKernel>& history) const
        {
            if(history.contains(kernelRun.toHash()))
            {
                bool found{false};
                recurse(tuneables, 0, kernelRun, history, found);
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
            if(!history.empty())
            {
                auto best = history.begin()->second;
                for(auto& run : history)
                {
                    if(run.second.metric.top() < best.metric.top())
                    {
                        best = run.second; // Update selectedRun to the run with the smaller time
                    }
                }
                toActive(kernelRun, best);
            }
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
