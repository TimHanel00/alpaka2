//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/interfaces/MetricInterface.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>
#include <alpaka/tune/utils/Random.hpp>

#if defined(strategy_bayesianOptimization)
#    include <alpaka/tune/interfaces/strategy_Impl/bayesionOptimizer.hpp>
#endif
#if defined(strategy_simulatedAnnealing)
#    include <alpaka/tune/interfaces/strategy_Impl/simA_oldStub.hpp>
#endif
namespace alpaka::tune::strategy
{


    struct randomSample
    {
        // template<typename T>
        // constexpr bool is_signed_type = std::is_signed_v<T>;
        //
        // template<typename T>
        // auto randomIdx(std::vector<T> const& valueList)
        // {
        //     if(valueList.empty())
        //     {
        //         throw std::runtime_error("randomIdx: valueList is empty");
        //     }
        //
        //     std::uniform_int_distribution<std::size_t> dis(0, valueList.size() - 1);
        //     return T{valueList[dis(RNG::get())]};
        // }
        //
        // template<typename VecT>
        // auto randomIdx(std::array<std::vector<typename VecT::type>, VecT::dim()> const& valueLists)
        // {
        //     VecT result;
        //     for(std::size_t d = 0; d < VecT::dim(); ++d)
        //     {
        //         if(valueLists[d].empty())
        //             throw std::runtime_error("randomIdx: empty valueList in dimension " + std::to_string(d));
        //
        //         std::uniform_int_distribution<std::size_t> dis(0, valueLists[d].size() - 1);
        //         result[d] = valueLists[d][dis(RNG::get())];
        //     }
        //     return result;
        // }
        // template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        // auto operator()(
        //     T_metricInterface& metricInterface, // the user specified metricInterface
        //     KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
        //     ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend Config
        //     EnvironmentState<T_Config>& environmentState) // contains global break criterias
        // {
        //     for_each(
        //         model.getUniformInterface(),
        //         [](auto& parameter)
        //         {
        //             auto val = randomIdx(parameter.getValues());
        //             parameter.value = val;
        //         });
        // };
    };

    /**
     *this is a exhaustive search method designed to support asymmetric index ranges and initial values that
     *might not even be on the range (meaning: (value-begin)%stride!=0 && (end-value)%stride!=0)
     *
     * */
    struct exhaustiveSearch
    {
        //
        //         template<auto N, typename T_Model>
        //         auto computeValueIndices(KernelTuningModelView<T_Model>& model)
        //         {
        //             using VecT = alpaka::Vec<std::size_t, N>;
        //             VecT idx;
        //
        //             for_each_enumerate(
        //                 model.getUniformInterface(),
        //                 [&](auto& t, std::size_t i)
        //                 {
        //                     auto& values = t.getValues();
        //                     using ValueT = std::decay_t<decltype(values[0])>;
        //                     if constexpr(std::is_arithmetic_v<ValueT>)
        //                     {
        //                         std::cout << std::endl;
        //                         auto it = std::find(values.begin(), values.end(), t.value[0]);
        //                         if(it == values.end())
        //                         {
        //                             std::cerr << "[ERROR] Value not found in candidate list for Tuneable[" << i <<
        //                             "]\n";
        //                             // there are for sure edge cases where this failsafe introduces some form of
        //                             inaccurate
        //                             // global state but atleast this is caught in the outer scope via a finite loop
        //                             for
        //                             // strategy calls
        //
        //                             it = values.begin();
        //                         }
        //                         idx[i] = std::distance(values.begin(), it);
        //                     }
        //                     else if constexpr(alpaka::concepts::Vector<std::decay_t<ValueT>>)
        //                     {
        //                         auto it = std::find_if(
        //                             values.begin(),
        //                             values.end(),
        //                             [&](auto const& v)
        //                             {
        //                                 auto cmp = v == t.value;
        //                                 bool match = allTrue(cmp);
        //                                 return match;
        //                             });
        //
        //                         if(it == values.end())
        //                         {
        //                             std::cerr << "[ERROR] No matching vector found for Tuneable[" << i << "]\n";
        //                             it = values.begin();
        //                             // there are for sure edge cases where this failsafe introduces some form of
        //                             inaccurate
        //                             // global state but atleast this is caught in the outer scope via a finite loop
        //                             for
        //                             // strategy calls
        //                         }
        //
        //                         idx[i] = std::distance(values.begin(), it);
        //                     }
        //                     else
        //                     {
        //                         static_assert(!std::is_same_v<ValueT, ValueT>, "Unsupported value type in exhaustive
        //                         search.");
        //                     }
        //                 });
        //             return idx;
        //         }
        //
        //         template<std::size_t N, typename T>
        //         constexpr std::size_t mapFromND(Vec<T, N> const& idx, Vec<T, N> const& dims)
        //         {
        //             std::size_t flat = 0;
        //             std::size_t mult = 1;
        //             for(std::size_t i = N; i-- > 0;)
        //             {
        //                 flat += idx[i] * mult;
        //                 mult *= dims[i];
        //             }
        //             return flat;
        //         }
        //
        //         std::vector<std::size_t> dimsVec;
        //         std::size_t total = 1;
        //         std::size_t stateCount = 0;
        //         std::size_t current1DImIndex = 0;
        //         bool init = false;
        // #define ExhaustiveSearchRandomInitialization
        // #ifdef ExhaustiveSearchRandomInitialization
        //         static constexpr bool randomInit = true;
        // #else
        //         static constexpr bool randomInit = false;
        // #endif
        //         template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        //         auto operator()(
        //             T_metricInterface& metricInterface, // the user specified metricInterface
        //             KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
        //             ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend
        //             Config EnvironmentState<T_Config>& environmentState) // contains global break criterias
        //         {
        //             using T_interface = decltype(model.getUniformInterface());
        //             constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<T_interface>>;
        //             using VecT = Vec<std::size_t, N>;
        //             if(!init)
        //             {
        //                 stateCount = 0;
        //                 if constexpr(randomInit)
        //                 {
        //                     randomSample{}(metricInterface, model, config_storage, environmentState);
        //                 }
        //
        //                 dimsVec.clear();
        //                 total = 1;
        //
        //                 for_each_enumerate(
        //                     model.getUniformInterface(),
        //                     [&](auto& t, std::size_t i)
        //                     {
        //                         std::size_t sz = t.getValues().size();
        //                         dimsVec.push_back(sz);
        //                         total *= sz;
        //                     });
        //
        //                 VecT idx = computeValueIndices<N>(model);
        //                 VecT dimsVecAsVec = convertVec<N>(dimsVec);
        //                 // since the first two configs where already gernerated, init and now this randomInit
        //                 current1DImIndex = mapFromND<N>(idx, dimsVecAsVec);
        //                 current1DImIndex = (current1DImIndex + 1) % total;
        //                 stateCount++;
        //                 init = true;
        //                 return;
        //             }
        //             if(stateCount >= total)
        //             {
        //                 environmentState.sessionFinished = true;
        //                 model.fromConfig(environmentState.getBestConfig());
        //                 return;
        //             }
        //             VecT nd = alpaka::mapToND(convertVec<N>(dimsVec), current1DImIndex);
        //             for_each_enumerate(
        //                 model.getUniformInterface(),
        //                 [&](auto& t, std::size_t i)
        //                 {
        //                     auto& vals = t.getValues();
        //                     if(nd[i] >= vals.size())
        //                     {
        //                         std::abort(); // Stop immediately
        //                     }
        //                     t.value = vals[nd[i]];
        //                 });
        //             stateCount++;
        //             current1DImIndex = (current1DImIndex + 1) % total;
        //         }
    };

    struct iterativeRefinement
    {
        // std::size_t m_numIterations = 5;
        // double_t resolution = 10;
        // iterativeRefinement() = default;
        //
        // explicit iterativeRefinement(std::size_t _numIterations) : m_numIterations(_numIterations)
        // {
        // }
        //
        // std::size_t curIteration = 0;
        //
        // template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        // auto operator()(
        //     T_metricInterface& metricInterface, // the user specified metricInterface
        //     KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
        //     ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend Config
        //     EnvironmentState<T_Config>& environmentState) // contains global break criterias
        // {
        //     exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);
        //
        //     auto config = model.toConfig();
        //     if(config_storage.contains(config))
        //     {
        //         exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);
        //     }
        //
        //     if(curIteration < m_numIterations)
        //     {
        //         if(config_storage.nrOfConfigs + 2 >= environmentState.maxConfigsTotal)
        //         {
        //             auto& best = config_storage.getOrCreate(environmentState.getBestConfig());
        //
        //             model.fromConfig(best);
        //
        //             std::apply(
        //                 [&]<typename... T>(T&... t)
        //                 {
        //                     (
        //                         [this]<typename U>(U& tune)
        //                         {
        //                             refineTuneableRange(tune, resolution);
        //                             tune.toRange();
        //                         }(t),
        //                         ...);
        //                 },
        //                 model.allTuneables());
        //
        //             curIteration++;
        //
        //             // alpaka::tune::recalculateMaxRuns(kernelRun);
        //             config_storage.nrOfConfigs = 0;
        //
        //             // Optional debug:
        //             // std::cout << "[Refinement] Running second exhaustive search...\n";
        //             // exhaustiveSearch<T_Metric>{}(tuneables, kernelRun, kernel_data);
        //         }
        //     }
        // };
    };

    struct randomSearch
    {
        auto operator()() const noexcept {};
        // template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        // auto operator()(
        //     T_metricInterface& metricInterface, // the user specified metricInterface
        //     KernelTuningModelView<T_TuningModel>& model,
        //     ConfigStorage<T_Config>&
        //         config_storage, // this already returns the Config for a specific kernel backend Config
        //     EnvironmentState<T_Config>& environmentState) // contains
        // {
        //     randomSample{}(metricInterface, model, config_storage, environmentState);
        //
        //     if(config_storage.contains(model.toConfig()))
        //     {
        //         exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);
        //     }
        // };
    };
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
