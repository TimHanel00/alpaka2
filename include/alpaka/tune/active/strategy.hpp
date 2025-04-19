//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include "alpaka/tune/IO/storageTypes.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <alpaka/tune/active/MetricInterface.hpp>

#include <random>
#include <vector>

template<typename T_history, typename T_kernelRun>
static bool kernelConfigChecked(T_history& history, T_kernelRun& kernel)
{
    if(!history.contains(kernel.toHash()))
        return false;
    if(history[kernel.toHash()].size() < getRunsPerConfig() || !history[kernel.toHash()].fullFlag)
        return false;
    return true;
}

namespace alpaka::tune::strategy
{
    class RNG
    {
    public:
        static std::mt19937& get()
        {
            static RNG instance;
            return instance.rng_;
        }

    private:
        RNG()
        {
            std::random_device rd;
            rng_ = std::mt19937(rd());
        }

        std::mt19937 rng_;
    };

    template<typename T>
    constexpr bool is_signed_type = std::is_signed<T>::value;

    template<typename T>
    T randomIdx(IdxRangeHandle<T> const& range, auto& value)
    {
        // Alias the vector type for the result.

        // Create a result vector.
        T result;
        // Assume that T_Begin has a static member T_dim (or use T_End::T_dim).

        // Set up a random number generator.
        // (Using static so that the generator is not re-seeded on every call.)
        // For each dimension, retrieve the minimum, maximum and stride.
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        assert(step != 0 && "Stride of Tuneable must be non-negative!");
        assert(value >= minVal && value <= maxVal && "Value of Tuneable is not in idxRange");
        using VecType = ALPAKA_TYPEOF(minVal);
        auto numStepsUp = (maxVal - VecType(value)) / step;
        if constexpr(is_signed_type<ALPAKA_TYPEOF(step)>)
        {
            numStepsUp = (maxVal - VecType(value)) / abs(step);
        }

        auto numStepsDown = (VecType(value) - minVal) / step;
        if constexpr(is_signed_type<ALPAKA_TYPEOF(step)>)
        {
            numStepsDown = (VecType(value) - minVal) / abs(step);
        }
        std::uniform_int_distribution<decltype(minVal)> dis(0, numStepsUp + numStepsDown);
        auto k = dis(RNG::get());
        if(k > numStepsUp)
        {
            std::uniform_int_distribution<decltype(minVal)> disD(0, numStepsDown);
            auto numStep = disD(RNG::get());
            result = value - (numStep * step);
        }
        else
        {
            std::uniform_int_distribution<decltype(minVal)> disU(0, numStepsUp);
            auto numStep = disU(RNG::get());
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
        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            T_tuneables&& tuneables,
            [[maybe_unused]] T_ActiveKernel& kernelRun,
            [[maybe_unused]] KernelData& kernel_data) const
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

    template<typename T_range, typename T_value>
    T_value randomNeighbour(T_range& range, T_value& value, bool& valid)
    {
        static std::random_device rd;
        bool validL = true;
        bool validH = true;
        auto lower = getNextLower(value, range, validL);
        auto higher = getNextgetNextUpper(value, range, validH);
        if(lower && higher)
        {
            std::uniform_int_distribution<std::size_t> dis(0, 1);
            auto k = dis(rd);
            if(k == 0)
                return lower;
            return higher;
        }
        if(lower)
            return lower;
        if(higher)
            return higher;
        valid = false;
        // has no neighbour
        return value;
    }

    namespace propabilityFunctions
    {
        struct Exponential
        {
            auto operator()(std::size_t distance, double_t temperature) const
            {
                return std::exp(-static_cast<double_t>(distance) / temperature);
            }
        };

        struct Normal
        {
            auto operator()(std::size_t distance, double_t temperature) const
            {
                double_t d = static_cast<double_t>(distance);
                return (1.0 / std::sqrt(2.0 * M_PI * temperature)) * std::exp(-d * d / (2.0 * temperature));
            }
        };

        struct Cauchy
        {
            auto operator()(std::size_t distance, double_t temperature) const
            {
                double_t d = static_cast<double_t>(distance);
                return (1.0 / M_PI) * (temperature / (d * d + temperature * temperature));
            }
        };

        struct StableHalf
        {
            auto operator()(std::size_t distance, double_t temperature) const
            {
                double_t d = static_cast<double_t>(distance);
                if(d == 0.0)
                    return 0.0;
                return (1.0 / std::sqrt(2.0 * M_PI * std::pow(d, 3))) * std::exp(-1.0 / (2.0 * d));
            }
        };
    } // namespace propabilityFunctions

    class Timing
    {
    };

    //@TODO move to different namespace
    template<typename T_Metric = alpaka::tune::Timing>
    struct bestRecorded
    {
        template<typename T_ActiveKernel>
        auto operator()(T_ActiveKernel& kernelRun, KernelData& history)
        {
            static std::unordered_map<std::string, std::vector<StorageKernelRun>> storeBestResults;
            // the first entry of each map is used to stay unique across several session instances (where for example
            // specifier could change)
            static std::unordered_map<std::string, std::string> bestResult;
            if(bestResult.contains(history.toHash()) && bestResult[history.toHash()] == kernelRun.toHash())
                return;
            if(!history.runs.empty())
            {
                auto best = history.runs.begin()->second;
                for(auto& entry : history.runs)
                {
                    StorageKernelRun& run = entry.second;
                    ::detail::Comparison res = run.compare(best);
                    switch(res)
                    {
                    case ::detail::Comparison::Greater:
                        best = MetricAdjust::aGTb<T_Metric>{}(run, best);
                        break;
                    case ::detail::Comparison::Less:
                        best = MetricAdjust::aLTb<T_Metric>{}(run, best);
                        break;
                    case ::detail::Comparison::Inconclusive:
                        if(run.getMetric<mean_t>().as<t_ns>() < best.getMetric<mean_t>().as<t_ns>())

                            best = run; // use mean as a tie-breaker in case statistical characteristics of the
                                        // distribution are similar
                        break;
                    default:
                        break;
                    }
                }
                bestResult[history.toHash()] = kernelRun.toHash();
                storeBestResults[history.toHash()].push_back(best);
                toActive(kernelRun, best);
                /**
                 *go again over all entries, now that we have identified one "best" configuration,
                 *this time write out "equal" entries according the kruskal wallis test
                 *this can be useful for later debugging or to write out a compact result list in production runs.
                 **/
                for(auto& entry : history.runs)
                {
                    StorageKernelRun& run = entry.second;
                    ::detail::Comparison res = run.compare(best);
                    switch(res)
                    {
                    case ::detail::Comparison::Inconclusive:
                        storeBestResults[history.toHash()].push_back(run);
                    default:
                        break;
                    }
                }
            }
        }
    };

    /**
     * extensible compare operator for certain metrics
     * */

    template<typename T_Metric = alpaka::tune::Timing>
    struct simulatedAnnealing
    {
        using T_propabilityFunction = propabilityFunctions::Exponential;
        static constexpr double T_final
            = 2.0; // magic Number that indicates the lower bound of the temperature used for simulated annealing

        double_t calcTemperature(auto const& maxRuns, auto currentRuns) const
        {
            auto currentRuns_local = std::max<std::size_t>(1, currentRuns);
            auto n0 = static_cast<double_t>(maxRuns) * 0.1; // log stabilizer prevent div by 0
            auto lambda = T_final * std::log(maxRuns + n0); // Tfinal*ln(N+n0)-> lamda is the the scaling
                                                            // factor of the temperature cooling
            auto result = lambda / std::log(static_cast<double_t>(currentRuns_local) * n0);
            std::cout << "[SA temp]: maxRuns" << maxRuns << " curRuns " << currentRuns_local
                      << " resulting temperature: " << result << std::endl;
            return result;
        }

        /*
         *TODO add genericOperator
         */
        bool acceptanceFunction(StorageKernelRun& metricNew, StorageKernelRun& metricOld, double const& temperature)
        {
            if(metricNew.getMetric<median_t>().as<t_ns>() < metricOld.getMetric<median_t>().as<t_ns>())
            {
                auto& preferred = MetricAdjust::aLTb<T_Metric>{}(metricNew, metricOld);
                if(&preferred == &metricNew)
                    return true;
            }

            double_t probability
                = std::exp(-(MetricAdjust::costDifference<T_Metric>{}(metricNew, metricOld) / temperature));
            std::uniform_int_distribution<std::size_t> dis(0, 1000);
            auto k = dis(RNG::get());
            if(k < probability * 1000)
            {
                // new solution will be accepted with this propability
                return true;
            }
            return false;
            // case the new metric is lower
        }

        /*
         * accept a already stored ParameterConfiguration with the likelyhood of the acceptance function
         */
        template<typename T_activeKernel>
        void acceptNewKernel(
            StorageKernelRun& oldKernel,
            StorageKernelRun& newKernel,
            T_activeKernel& activeKernel,
            double_t temperature)
        {
            if(acceptanceFunction(newKernel, oldKernel, temperature))
            {
                toActive(activeKernel, newKernel);
            }
            else
            {
                toActive(activeKernel, oldKernel);
            }
        }

        /*
         * applys a heuristic on a parameter to select the neighboorhood for each parameter individually based on a
         * propability function that is affected by the cooling rate,
         * inspired by:
         * https://citeseerx.ist.psu.edu/document?doi=c8dcf69dbc8c750b2db5f16e1e737017efd7dd4a&repid=rep1&type=pdf&utm_source=chatgpt.com
         * in the paper they use the hamming distance between two configurations which would include all parameters
         * but applying it per parameter simplifies the computation and algorithm
         */
        std::size_t applyProbabilityFunction(auto currentValue, auto const& range, double_t temperature)
        {
            std::size_t backwardSteps = (currentValue - range.m_begin) / range.m_stride;
            std::size_t forwardSteps = (range.m_end - currentValue) / range.m_stride;
            std::vector<double> weights(backwardSteps + forwardSteps + 1);
            double total = 0.0;

            for(std::size_t i = 0; i <= backwardSteps; ++i)
            {
                weights[backwardSteps - i] = T_propabilityFunction{}(i, temperature); // backward
                total += weights[backwardSteps - i];
            }
            for(std::size_t i = 1; i <= forwardSteps; ++i)
            {
                weights[backwardSteps + i] = T_propabilityFunction{}(i, temperature); // forward
                total += weights[backwardSteps + i];
            }

            for(auto& w : weights)
                w /= total;
            std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
            std::size_t sampledIndex = dist(RNG::get());
            if(sampledIndex < backwardSteps)
            {
                std::size_t stepsBack = backwardSteps - sampledIndex;
                return currentValue - stepsBack * range.m_stride; // go backwards
            }
            std::size_t stepsForward = sampledIndex - backwardSteps;
            return currentValue + stepsForward * range.m_stride; // go forwards
        }

#define SimA_MaxCachedSteps 100

        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(T_tuneables&& tuneables, T_ActiveKernel& kernelRun, KernelData& kernel_data)
        {
            auto& history = kernel_data.runs;
            auto& state = kernelRun.m_strategyState;
            if(state.runs >= SimA_MaxCachedSteps)
            {
                std::cout << " selecting best config due to SimA steps exceeded" << std::endl;
                alpaka::tune::strategy::bestRecorded{}(kernelRun, kernel_data);
                return;
            }
            state.runs = 0; // reset
            auto maxRuns = getMaxRuns();
            if(!kernelConfigChecked(history, kernelRun))
            {
                return;
                // we havent yet timed this parameter config often enough to make a educated guess on its performance
                // therefore we return with the  -- this check might be redundant (TODO check if redundant)
            };
            // this means we evaluated the kernelRun activeKernel well enough
            if(!state.oldKernelHash.empty())
            {
                acceptNewKernel(
                    history[state.oldKernelHash],
                    history[kernelRun.toHash()],
                    kernelRun,
                    state.temperature); // we might switch to the latest config nevertheless
                // we accept the "new" config always if its better and with a propability of e^(-new+old)/temp)
                // if its worse (if lower is better)
            }


            state.temperature = calcTemperature(getMaxRuns(), history.size()); // assign new temperature
            while(kernelConfigChecked(history, kernelRun) && state.runs < SimA_MaxCachedSteps)
            {
                std::string oldKernelHash = kernelRun.toHash();
                for_each(
                    tuneables,
                    [state, this](auto& parameter)
                    {
                        auto newVal = applyProbabilityFunction(parameter.value, parameter.idxRange, state.temperature);
                        parameter.value = newVal;
                    });
                if(kernelRun.toHash() != state.oldKernelHash)
                {
                    state.oldKernelHash = oldKernelHash;
                }
                if(kernelConfigChecked(history, kernelRun))
                {
                    // we run in this case if the newly found config was already cached (evaluated enough)
                    // so we can decide directly if we want to go there
                    // we accept the new found config always if its better and with a propability of e^(-new+old)/temp)
                    // if its worse (if lower is better)
                    acceptNewKernel(
                        history[state.oldKernelHash],
                        history[kernelRun.toHash()],
                        kernelRun,
                        state.temperature);
                    ++state.runs;
                    // here we also count if we accept equal configs.
                }
                else
                {
                    // kernel not yet in history, therefore we have no evalutation of the new parameters and take them
                    return;
                }
            }


            // if we already have been to that config we still jump there with the propability function but we go to
            // the next config afterwards
        }
    };

    /**
     *this is a exhaustive search method designed to support asymmetric index ranges and initial values that
     *might not even be on the range (meaning: (value-begin)%stride!=0 && (end-value)%stride!=0)
     *
     * */
    template<typename T_Metric = alpaka::tune::Timing>
    struct exhaustiveSearch
    {
        template<typename Tuple, typename T_ActiveKernel, typename StorageKernel>
        void recurse(
            Tuple& tuneables,
            std::size_t dim,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, StorageKernel>& history,
            bool& found)
        {
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
            auto hash = kernelRun.toHash();
            if(found || !history.contains(hash) || history[hash].nr_runs < getRunsPerConfig())
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

        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(T_tuneables&& tuneables, T_ActiveKernel& kernelRun, KernelData& kernel_data)
        {
            auto& history = kernel_data.runs;
            if(history.contains(kernelRun.toHash()))
            {
                bool found{false};
                recurse(tuneables, 0, kernelRun, history, found);
            }
        }
    };

    template<typename T_Metric = alpaka::tune::Timing>
    struct randomSearch
    {
        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(T_tuneables&& tuneables, T_ActiveKernel& kernelRun, KernelData& kernel_data)
        {
            auto& history = kernel_data.runs;
            randomSample{}(tuneables, kernelRun, kernel_data);
            if(history.contains(kernelRun.toHash()))
            {
                exhaustiveSearch<T_Metric>{}(tuneables, kernelRun, kernel_data);
            }
        };
    };

    template<typename T_Metric = alpaka::tune::Timing>
    struct initialValues
    {
        template<typename tuneables, typename T_KernelRun, typename KernelRun>
        auto operator()(
            std::vector<std::shared_ptr<tuneables>>& tuningParameters,
            T_KernelRun& kernelRun,
            std::unordered_map<std::string, KernelRun>& history)
        {
            return tuningParameters;
        }
    };
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
