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
    constexpr bool is_signed_type = std::is_signed_v<T>;

    template<typename T>
    auto randomIdx(IdxRangeHandle<T> const& range, auto& value)
    {
        typename utils::toRTime<T>::get result;

        // Get the range values.
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        // std::cout << "[DEBUG]  maxVal" << maxVal.toString() << std::endl;
        // std::cout << "[DEBUG]  value" << value.toString() << std::endl;
        //  Step-by-step debug output
        auto diff = maxVal - value;
        // std::cout << "[DEBUG] (maxVal - value): " << diff.toString() << std::endl;

        auto absStep = utils::abs(step);
        // std::cout << "[DEBUG] utils::abs(step): " << absStep.toString() << std::endl;

        auto div = diff / absStep;
        // std::cout << "[DEBUG] (maxVal - value) / utils::abs(step): " << div.toString() << std::endl;

        auto numStepsUp = utils::min_element(div);
        // std::cout << "[DEBUG] utils::min_element((maxVal - value) / utils::abs(step)): " << numStepsUp << std::endl;

        auto numStepsDown = utils::min_element((value - minVal) / utils::abs(step));
        std::uniform_int_distribution<typename T::type> dis(0, numStepsUp + numStepsDown);
        auto k = dis(RNG::get());

        if(k > numStepsUp)
        {
            std::uniform_int_distribution<typename T::type> disD(0, numStepsDown);
            auto numStep = disD(RNG::get());
            result = value - (numStep * step);
        }
        else
        {
            std::uniform_int_distribution<typename T::type> disU(0, numStepsUp);
            auto numStep = disU(RNG::get());
            result = value + (numStep * step);
        }

        return result;
    }

    template<typename T, typename T_Ref>
    T getNextUpper(T const& value, IdxRangeHandle<T_Ref> const& range, bool& valid)
    {
        // Create a result vector.
        T result = value;
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        result = (result + step);

        if(utils::anyTrue(result < minVal) || utils::anyTrue(result > maxVal))
        {
            valid = false;
            return maxVal;
        }
        return result;
    }

    template<typename T, typename T_Ref>
    T getNextLower(T const& value, IdxRangeHandle<T_Ref> const& range, bool& valid)
    {
        // Create a result vector.
        T result = value;
        auto minVal = range.m_begin;
        auto maxVal = range.m_end;
        auto step = range.m_stride;
        result = (result - step);
        if(utils::anyTrue(result < minVal) || utils::anyTrue(result > maxVal))
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
            concepts::MetricInterface auto& metric_interface,
            T_tuneables&& tuneables,
            [[maybe_unused]] T_ActiveKernel& kernelRun,
            [[maybe_unused]] KernelData& kernel_data) const
        {
            for_each(
                tuneables,
                [](auto& parameter)
                {
                    auto val = randomIdx(parameter.idxRange, parameter.value);
                    parameter.value = val;
                });
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
        auto higher = getNextUpper(value, range, validH);
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
    struct bestRecorded
    {
        template<typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_ActiveKernel& kernelRun,
            KernelData& history)
        {
            using T_metricInterface = std::remove_cvref_t<decltype(metricInterface)>;
            static std::unordered_map<std::string, std::vector<StorageKernelRun>> storeBestResults;
            static std::unordered_map<std::string, std::string> bestResult;
            if(bestResult.contains(history.toHash()) && bestResult[history.toHash()] == kernelRun.toHash())
            {
                return;
            }
            if(!history.runs.empty())
            {
                StorageKernelRun best = history.runs.begin()->second;

                for(auto& entry : history.runs)
                {
                    StorageKernelRun& run = entry.second;
                    ::Comparison res = run.compare(best);

                    switch(res)
                    {
                    case ::Comparison::Greater:
                        best = aGTb<T_metricInterface>{}(run, best);
                        break;

                    case ::Comparison::Less:
                        best = aLTb<T_metricInterface>{}(run, best);
                        break;

                    case ::Comparison::Inconclusive:
                        {
                            auto runMean = run.getMetric<mean_t>().as<t_ns>();
                            auto bestMean = best.getMetric<mean_t>().as<t_ns>();
                            if(runMean < bestMean)
                            {
                                best = run;
                            }
                            break;
                        }

                    default:
                        break;
                    }
                }
                bestResult[history.toHash()] = kernelRun.toHash();
                storeBestResults[history.toHash()].push_back(best);
                toActive(kernelRun, best);

                for(auto& entry : history.runs)
                {
                    StorageKernelRun& run = entry.second;
                    ::Comparison res = run.compare(best);
                    if(res == ::Comparison::Inconclusive)
                    {
                        storeBestResults[history.toHash()].push_back(run);
                    }
                }
            }
        }
    };

    /**
     * extensible compare operator for certain metrics
     * */

    struct simulatedAnnealing
    {
        using T_propabilityFunction = propabilityFunctions::Exponential;
        static constexpr double T_final
            = 5.0; // magic Number that indicates the lower bound of the temperature used for simulated annealing

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

        template<typename T_Metric>
        bool acceptWorseSolution(StorageKernelRun& metricNew, StorageKernelRun& metricOld, double const& temperature)
        {
            double_t probability = std::exp(
                -(alpaka::tune::strategy::SimulatedAnnealing::costDifference<T_Metric>{}(metricNew, metricOld)
                  / temperature));
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
         *TODO add genericOperator
         */
        template<typename T_Metric>
        bool acceptanceFunction(StorageKernelRun& metricNew, StorageKernelRun& metricOld, double const& temperature)
        {
            auto res = metricNew.compare(metricOld);
            switch(res)
            {
            case ::Comparison::Greater:
                auto& preferred = aLTb<T_Metric>{}(metricNew, metricOld);
                if(&preferred == &metricNew)
                    return true;
                return acceptWorseSolution<T_Metric>(metricNew, metricOld, temperature);
            case ::Comparison::Less:
                preferred = aGTb<T_Metric>{}(metricNew, metricOld);
                if(&preferred == &metricNew)
                    return true;
                return acceptWorseSolution<T_Metric>(metricNew, metricOld, temperature);
            case ::Comparison::Inconclusive:
                return true; // encourage exploration
            case ::Comparison::Dummy:
                return false;
            default:;
            }


            return false;
        }

        /*
         * accept a already stored ParameterConfiguration with the likelyhood of the acceptance function
         */
        template<typename T_Metric, typename T_activeKernel>
        void acceptNewKernel(
            StorageKernelRun& oldKernel,
            StorageKernelRun& newKernel,
            T_activeKernel& activeKernel,
            double_t temperature)
        {
            if(acceptanceFunction<T_Metric>(newKernel, oldKernel, temperature))
            {
                // toActive(activeKernel, newKernel); -> we dont have to do anything since ActiveKernel is already in
                // the newKernel config
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
        auto applyProbabilityFunction(auto currentValue, auto const& range, double_t temperature)
        {
            using type = typename decltype(currentValue)::type; // this is a primitive type
            auto backwardSteps
                = utils::min_element((currentValue - range.m_begin) / range.m_stride); // this is a n dim vector dim>=1
            auto forwardSteps
                = utils::min_element((range.m_end - currentValue) / range.m_stride); // this is a n dim vector dim>=1

            std::vector<double> weights(backwardSteps + forwardSteps + 1);
            double total = 0.0;

            for(type i = 0; i <= backwardSteps; ++i)
            {
                weights[backwardSteps - i] = T_propabilityFunction{}(i, temperature); // backward
                total += weights[backwardSteps - i];
            }
            for(type i = 1; i <= forwardSteps; ++i)
            {
                weights[backwardSteps + i] = T_propabilityFunction{}(i, temperature); // forward
                total += weights[backwardSteps + i];
            }

            for(auto& w : weights)
                w /= total;
            std::discrete_distribution<type> dist(weights.begin(), weights.end());
            type sampledIndex = dist(RNG::get());
            if(sampledIndex < backwardSteps)
            {
                type stepsBack = backwardSteps - sampledIndex;
                auto backwardsStepVec = utils::toRTime<decltype(currentValue)>::get::all(stepsBack);

                return currentValue - backwardsStepVec * range.m_stride; // this is a correct calculation if stepsBack
                                                                         // has the same dimension as curVal
            }
            type stepsForward = sampledIndex - backwardSteps;
            auto forwardsStepVec = utils::toRTime<decltype(currentValue)>::get::all(stepsForward);
            return currentValue + forwardsStepVec * range.m_stride; // this is a correct calculation if stepsBack has
                                                                    // the same dimension as curVal
        }

#define SimA_MaxCachedSteps 300

        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data)
        {
            using T_Metric = std::remove_cvref<decltype(metricInterface)>;
            auto& history = kernel_data.runs;
            auto& state = kernelRun.m_strategyState;
            if(state.runs >= std::max(
                   std::min(getMaxRuns(), static_cast<std::size_t>(SimA_MaxCachedSteps * 20)),
                   static_cast<std::size_t>(
                       SimA_MaxCachedSteps))) // clamp steps between
                                              // SimA_MaxCachedSteps<state.runs<SimA_MaxCachedSteps*20
            {
                std::cout << " selecting best config due to SimA steps exceeded" << std::endl;
                kernelRun.m_strategyState.done = true;
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
                acceptNewKernel<T_Metric>(
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
                    // we m_run in this case if the newly found config was already cached (evaluated enough)
                    // so we can decide directly if we want to go there
                    // we accept the new found config always if its better and with a propability of e^(-new+old)/temp)
                    // if its worse (if lower is better)
                    acceptNewKernel<T_Metric>(
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
                    auto oldValue = utils::toRT(param.value); // creates a temporary vector from a RefStorage Vector

                    // Try all values in the index range for this parameter
                    {
                        bool valid = true;
                        auto value = utils::toRT(param.value);
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
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data)
        {
            auto& history = kernel_data.runs;
            if(history.contains(kernelRun.toHash()))
            {
                bool found{false};
                recurse(tuneables, 0, kernelRun, history, found);
                if(!found)
                {
                    // fallback incase we found no new or still usable config it indicates that we switch to bestConfig
                    kernel_data.nrOfConfigs = kernelRun.maxRuns;
                }
                std::cout << " was valid: " << found << std::endl;
            }
        }
    };

    struct Refinement
    {
        // Primary template — default case
        template<std::size_t N, typename Enable = void>
        struct Op;

        // Default when no specialization is registered
        template<std::size_t N>
        struct Op<N>
        {
            void operator()(alpaka::concepts::tuneable auto& tune, std::size_t resolution)
            {
                using T_tune = std::remove_cvref_t<decltype(tune)>;
                using T_range = decltype(tune.idxRange);
                using T_Vec = typename T_tune::ValueType;
                using valType = std::remove_reference_t<decltype(tune.value[0])>;
                auto range = (tune.idxRange.m_end - tune.idxRange.m_begin);
                auto nrSteps = range / tune.idxRange.m_stride;
                auto percentageDeviation = (1.0 / static_cast<double_t>(resolution));
                for(int i = 0; i < alpaka::getDim(T_Vec{}); i++)
                {
                    tune.idxRange.m_begin[i] = std::max(
                        static_cast<valType>(tune.idxRange.m_begin[i]),
                        static_cast<valType>(tune.value[i] - percentageDeviation * range));
                    tune.idxRange.m_end[i] = std::min(
                        tune.idxRange.m_end[i],
                        static_cast<valType>(tune.value[i] + percentageDeviation * range));

                    tune.idxRange.m_stride[i]
                        = static_cast<valType>((tune.idxRange.m_end[i] - tune.idxRange.m_begin[i]) / nrSteps[i]);
                    if(std::is_integral_v<valType> && tune.idxRange.m_stride[i] == valType(0))
                    {
                        tune.idxRange.m_stride[i] = 1;
                    }
                }
                std::cout << " new range for: 0 " << tune.idxRange.m_begin.toString()
                          << " end: " << tune.idxRange.m_end.toString()
                          << " stride: " << tune.idxRange.m_stride.toString() << std::endl;
            }
        };
    };

    template<>
    struct Refinement::Op<static_cast<std::size_t>(alpaka::tune::frameTune::numBlocks)>
    {
        void operator()(alpaka::concepts::tuneable auto& tune, std::size_t resolution)
        {
            // example special refinenemt for numBlocks ( in this case doesnt get changed on refinement update cycle)
            using T_tune = std::remove_cvref_t<decltype(tune)>;
            using T_range = decltype(tune.idxRange);
            using T_Vec = typename T_tune::ValueType;
            using valType = std::remove_reference_t<decltype(tune.value[0])>;
            std::cout << " new range for: 0 " << tune.idxRange.m_begin.toString()
                      << " end: " << tune.idxRange.m_end.toString() << " stride: " << tune.idxRange.m_stride.toString()
                      << std::endl;
        }
    };

    template<typename T_Tuneable>
    void refineTuneableRange(T_Tuneable& tune, std::size_t resolution)

    {
        using tuneableType = std::remove_cvref_t<T_Tuneable>;
        constexpr auto id = tuneableType::tag;
        Refinement::Op<id>{}(tune, resolution);
    }

    struct iterativeRefinement
    {
        std::size_t m_numIterations = 5;
        double_t resolution = 10;
        iterativeRefinement() = default;

        explicit iterativeRefinement(std::size_t _numIterations) : m_numIterations(_numIterations)
        {
        }

        std::size_t curIteration = 0;

        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data)
        {
            exhaustiveSearch{}(metricInterface, tuneables, kernelRun, kernel_data);

            auto kernelHash = kernelRun.toHash();
            if(kernel_data.runs.contains(kernelHash))
            {
                exhaustiveSearch{}(metricInterface, tuneables, kernelRun, kernel_data);
            }

            if(curIteration < m_numIterations)
            {
                if(kernel_data.nrOfConfigs + 2 >= kernelRun.maxRuns)
                {
                    bestRecorded{}(metricInterface, kernelRun, kernel_data);

                    std::apply(
                        [&]<typename... T>(T&... t)
                        {
                            (
                                [this]<typename U>(U& tune)
                                {
                                    refineTuneableRange(tune, resolution);
                                    tune.toRange();
                                }(t),
                                ...);
                        },
                        kernelRun.allTuneables());

                    curIteration++;

                    alpaka::tune::recalculateMaxRuns(kernelRun);
                    kernel_data.nrOfConfigs = 0;

                    // Optional debug:
                    // std::cout << "[Refinement] Running second exhaustive search...\n";
                    // exhaustiveSearch<T_Metric>{}(tuneables, kernelRun, kernel_data);
                }
            }
        };
    };

    struct randomSearch
    {
        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data)
        {
            using T_Metric = std::remove_cvref_t<decltype(metricInterface)>;
            auto& history = kernel_data.runs;
            /*
            std::cout << " before random Sample: " << std::endl;
            std::apply(
                [](auto&... elem)
                {
                    ((std::cout << " value: " << elem.value.toString() << " begin: "
                                << elem.idxRange.m_begin.toString() << " end: " << elem.idxRange.m_end.toString()
                                << " stride: " << elem.idxRange.m_stride.toString() << std::endl),
                     ...);
                },
                tuneables);*/
            std::cout << "before " << kernelRun.toHash() << std::endl;
            randomSample{}(metricInterface, tuneables, kernelRun, kernel_data);
            /*
            std::cout << " after random Sample: " << std::endl;
            std::apply(
                [](auto&... elem)
                {
                    ((std::cout << " value: " << elem.value.toString() << " begin: "
                                << elem.idxRange.m_begin.toString() << " end: " << elem.idxRange.m_end.toString()
                                << " stride: " << elem.idxRange.m_stride.toString() << std::endl),
                     ...);
                },
                tuneables);
            std::cout << " finish " << std::endl;
            */
            std::cout << "after " << kernelRun.toHash() << std::endl;
            if(history.contains(kernelRun.toHash()))
            {
                exhaustiveSearch{}(metricInterface, tuneables, kernelRun, kernel_data);
            }
        };
    };
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
