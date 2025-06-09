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
            [[maybe_unused]] KernelData& kernel_data,
            EnvironmentState& state) const
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

    /**
     * extensible compare operator for certain metrics
     * */

    struct simulatedAnnealing
    {
        using T_propabilityFunction = propabilityFunctions::Exponential;
        static constexpr double T_init = 100.0;
        static constexpr double T_final
            = 5.0; // magic Number that indicates the lower bound of the temperature used for simulated annealing

        double_t temperature = T_init; // class member

        double_t calcTemperature(auto const& maxRuns, auto currentRuns) const
        {
            auto currentRuns_local = std::max<std::size_t>(1, currentRuns);
            auto n0 = static_cast<double_t>(maxRuns) * 0.1; // log stabilizer prevent div by 0
            auto lambda = T_final * std::log(maxRuns + n0); // Tfinal*ln(N+n0)-> lamda is the the scaling
                                                            // factor of the temperature cooling
            auto result = lambda / std::log(static_cast<double_t>(currentRuns_local) * n0);
            return std::clamp(result, T_final, T_init);
        }

        template<typename T_Metric>
        bool acceptWorseSolution(StorageKernelRun const& cand, StorageKernelRun const& cur, double temperature)
        {
            double delta = alpaka::tune::strategy::SimulatedAnnealing::costDifference<T_Metric>{}(cand, cur);

            if(delta <= 0.0)
                return true; // better or equal → accept

            double prob = std::exp(-delta / temperature);

            static thread_local std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            return dist(rng) < prob;
        }

        /*
         *TODO add genericOperator
         */
        template<typename T_Metric>
        bool acceptanceFunction(StorageKernelRun const& cand, StorageKernelRun const& cur, double temperature)
        {
            switch(cand.compare(cur))
            {
            case ::Comparison::Greater:
                {
                    auto const& preferred = aGTb<T_Metric>{}(cand, cur);
                    return (&preferred == &cand) || acceptWorseSolution<T_Metric>(cand, cur, temperature);
                }
            case ::Comparison::Less:
                {
                    auto const& preferred = aLTb<T_Metric>{}(cand, cur);
                    return (&preferred == &cand) || acceptWorseSolution<T_Metric>(cand, cur, temperature);
                }
            case ::Comparison::Inconclusive:
                return true; // always explore
            default:
                return false;
            }
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
                toActive(activeKernel, newKernel);
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
        std::string previousValidKernel = "";
        std::size_t currentRuns = 0;

        template<typename T_KernelRun>
        inline bool existsInHistory(T_KernelRun& run, KernelData& kernel_data)
        {
            if(!kernel_data.runs.contains(run.toHash()))
                return false;
            return true;
        }

        inline bool acceptWorseForEdgeCases(float temperature)
        {
            if(temperature <= 0.0f)
                return false; // never accept worse if temperature is zero or below

            // Generate acceptance probability in [0, 1)
            double_t probability = std::exp(-4.0 / temperature); // constant "cost" of 1

            std::uniform_real_distribution<double_t> dist(0.0, 1.0);
            return dist(RNG::get()) < probability;
        }

        template<typename T_KernelRun>
        bool handleInvalidCases(
            StorageKernelRun& current,
            StorageKernelRun& worse,
            T_KernelRun& run,
            float temperature)
        {
            if(worse.state == StorageKernelRun::State::Dummy)
            {
                if(acceptWorseForEdgeCases(temperature))
                {
                    toActive(run, worse);
                }
                else
                {
                    toActive(run, current);
                }
                return true;
            }
            if(current.state == StorageKernelRun::State::Dummy)
            {
                if(acceptWorseForEdgeCases(temperature))
                {
                    toActive(run, current);
                }
                else
                {
                    toActive(run, worse);
                }
                return true;
            }
            return false;
        }

        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data,
            EnvironmentState& env_state)
        {
            using T_Metric = std::remove_cvref<decltype(metricInterface)>;
            auto& history = kernel_data.runs;
            StorageKernelRun& oldRun = kernel_data[kernelRun.toHash()];
            std::string oldHash = kernelRun.toHash();
            currentRuns = 0;
            temperature = calcTemperature(env_state.maxConfigsTotal, env_state.numberOfCheckedConfigs);
            while(currentRuns < SimA_MaxCachedSteps)
            {
                // calculateNewTemperature
                oldHash = kernelRun.toHash();
                for_each(
                    tuneables,
                    [this](auto& parameter)
                    {
                        auto newVal = applyProbabilityFunction(parameter.value, parameter.idxRange, temperature);
                        parameter.value = newVal;
                    });
                if(kernelRun.toHash() == oldHash)
                {
                    currentRuns++;
                    continue;
                }
                if(!existsInHistory(kernelRun, kernel_data))
                {
                    std::cout << " new config found! " << std::endl;
                    return;
                }
                StorageKernelRun& curRun = kernel_data[kernelRun.toHash()];
                oldRun = kernel_data.runs[oldHash];
                if(handleInvalidCases<T_Metric>(curRun, oldRun, kernelRun, temperature))
                {
                    currentRuns++;
                    continue;
                }
                // we m_run in this case if the newly found config was already cached (evaluated enough)
                // so we can decide directly if we want to go there
                // we accept the new found config always if its better and with a propability of
                // e^(-new+old)/temp) if its worse (if lower is better)
                acceptNewKernel<T_Metric>(oldRun, curRun, kernelRun, temperature);
                ++currentRuns;
                // here we also count if we accept equal configs.
            }
        }

        // if we already have been to that config we still jump there with the propability function but we go to
        // the next config afterwards
    };

    ;

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
            if(found || !history.contains(hash))
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

        bool init = false;
#ifdef ExhaustiveSearchRandomInitialization
        constexpr bool randomInit = true;
#else
        constexpr bool randomInit = false;
#endif
        template<typename T_tuneables, typename T_ActiveKernel>
        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            T_tuneables&& tuneables,
            T_ActiveKernel& kernelRun,
            KernelData& kernel_data,
            EnvironmentState& state)
        {
            auto& history = kernel_data.runs;
            if constexpr(randomInit)
            {
                if(!init)
                {
                    randomSample{}(metricInterface, tuneables, kernelRun, kernel_data, state);
                    init = true; // only the first time this strategy is called in the current context
                }
            }

            if(history.contains(kernelRun.toHash()))
            {
                bool found{false};
                recurse(tuneables, 0, kernelRun, history, found);
                if(!found)
                {
                    std::cout << " did not find any new tuneable use best recorded run now. " << std::endl;

                    // fallback incase we found no new or still usable config it indicates that we switch to bestConfig
                    state.sessionFinished = true;
                }
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

        auto operator()(
            concepts::MetricInterface auto& metricInterface,
            auto&& tuneables,
            auto& kernelRun,
            KernelData& kernel_data,
            EnvironmentState& state)
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
                    StorageKernelRun& best = kernel_data.runs[state.bestConfig.toHash()];
                    toActive(kernelRun, best);

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
            KernelData& kernel_data,
            EnvironmentState& state)
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
