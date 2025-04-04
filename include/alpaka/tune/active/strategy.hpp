//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include "alpaka/tune/IO/storageTypes.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <random>
#include <vector>

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

    /**
     * extensible compare operator for certain metrics
     * */
    struct MetricAdjust
    {
        template<typename T_Metric>
        struct best
        {
            T_Metric operator()(T_Metric const& a, T_Metric const& b) const
            {
                // Default: assume higher is better
                return (a < b) ? a : b;
            }
        };

        template<typename T_Metric>
        struct costDifference
        {
            T_Metric operator()(T_Metric const& a, T_Metric const& b) const
            {
                // Default: assume higher is better
                return a - b;
            }
        };
    };

    /*
     * since time is currently no defined interface this is used to compare time metrics
     * */
    template<>
struct MetricAdjust::best<double_t>
    {
        double_t operator()(double_t const& a, double_t const& b) const
        {
            return (a < b) ? a : b;
        }
    };

    // Specialize costDifference<double_t>
    template<>
    struct MetricAdjust::costDifference<double_t>
    {
        double_t operator()(double_t const& old_, double_t const& new_) const
        {
            return old_ - new_;
        }
    };


    struct simulatedAnnealing
    {
        using T_propabilityFunction = propabilityFunctions::Exponential;
        static constexpr double T_final
            = 0.1; // magic Number that indicates the lower bound of the temperature used for simulated annealing

        double_t calcTemperature(auto const& maxRuns, auto currentRuns) const
        {
            auto n0 = static_cast<double_t>(maxRuns) * 0.1; // log stabilizer prevent div by 0
            auto lambda = T_final * std::log(maxRuns + n0); // Tfinal*ln(N+n0)-> lamda is the the scaling
                                                            // factor of the temperature cooling
            return lambda / std::log(static_cast<double_t>(currentRuns) * n0);
        }

        /*
         *TODO add genericOperator
         */
        template<typename T_Metric>
        bool acceptanceFunction(T_Metric const& metricNew, T_Metric const& metricOld, double const& temperature)
        {
            if(metricNew == MetricAdjust::best<ALPAKA_TYPEOF(metricNew)>{}(metricNew, metricOld))
                return true;

            double_t probability = std::exp(
                -(MetricAdjust::costDifference<ALPAKA_TYPEOF(metricNew)>{}(metricNew, metricOld) / temperature));
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
        template<typename T_storageKernel, typename T_activeKernel>
        void acceptNewKernel(T_storageKernel& storekernel, T_activeKernel& activeKernel, double_t temperature)
        {
            if(acceptanceFunction(storekernel.metric, activeKernel.metric, temperature))
            {
                toActive(activeKernel, storekernel);
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
            std::size_t maxSteps = (range.m_end - currentValue) / range.m_stride;
            std::vector<double> weights(maxSteps + 1);
            double total = 0.0;

            for(std::size_t distance = 0; distance <= maxSteps; ++distance)
            {
                weights[distance] = T_propabilityFunction{}(distance, temperature);
                total += weights[distance];
            }

            for(auto& w : weights)
                w /= total;

            // Sample d from this distribution
            std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
            std::size_t d = dist(RNG::get());

            return currentValue + d * range.m_stride;
        }

        template<typename T_tuneables, typename T_ActiveKernel, typename storageKernel>
        auto operator()(
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            std::unordered_map<std::string, storageKernel>& history) const
        {
            auto& state = kernelRun.m_strategyState;
            auto maxRuns = getMaxRuns();
            if(!history.contains(kernelRun.toHash()))
            {
                state.runs = 1;
                return;
            }
            state.temperatur = calcTemperature(maxRuns, history.size()); // assign new temperature
            int checkOverlow = 0;
            while(history.contains(kernelRun.toHash()) && checkOverlow < 100)
            {
                for_each(
                    tuneables,
                    [state](auto& parameter)
                    {
                        parameter.value
                            = applyProbabilityFunction(parameter.value, parameter.idxRange, state.temperatur);
                    });
                if(history.contains(kernelRun.toHash()))
                {
                    acceptNewKernel(history[kernelRun.toHash()], kernelRun);
                    if(history[kernelRun.toHash()].metric)
                        toActive(kernelRun, history[kernelRun.toHash()]);
                    ++state.runs;
                }
                checkOverlow++;
            }
            if(checkOverlow > 99)
            {
                std::cout << " Simulated Annealing: no more valid config found for the last 100 iterations"
                          << std::endl;
                getMaxRuns(1);
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
