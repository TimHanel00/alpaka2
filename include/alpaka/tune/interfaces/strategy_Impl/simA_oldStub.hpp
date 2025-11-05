//
// Created by tim on 15.10.25.
//

#ifndef SIMA_OLDSTUB_H
#define SIMA_OLDSTUB_H

namespace alpaka::tune::strategy
{
    struct simulatedAnnealing
    {
        //         template<typename T_range, typename T_value>
        // T_value randomNeighbour(T_range& range, T_value& value, bool& valid)
        //         {
        //             static std::random_device rd;
        //             bool validL = true;
        //             bool validH = true;
        //             auto lower = getNextLower(value, range, validL);
        //             auto higher = getNextUpper(value, range, validH);
        //             if(lower && higher)
        //             {
        //                 std::uniform_int_distribution<std::size_t> dis(0, 1);
        //                 auto k = dis(rd);
        //                 if(k == 0)
        //                     return lower;
        //                 return higher;
        //             }
        //             if(lower)
        //                 return lower;
        //             if(higher)
        //                 return higher;
        //             valid = false;
        //             // has no neighbour
        //             return value;
        //         }
        // namespace propabilityFunctions
        // {
        //     struct Exponential
        //     {
        //         double_t operator()(std::size_t distance, double_t temperature) const
        //         {
        //             return std::exp(-static_cast<double_t>(distance) / temperature);
        //         }
        //     };
        //
        //     struct Normal
        //     {
        //         double_t operator()(std::size_t distance, double_t temperature) const
        //         {
        //             double_t d = static_cast<double_t>(distance);
        //             return (1.0 / std::sqrt(2.0 * M_PI * temperature)) * std::exp(-d * d / (2.0 * temperature));
        //         }
        //     };
        //
        //     struct Cauchy
        //     {
        //         double_t operator()(std::size_t distance, double_t temperature) const
        //         {
        //             double_t d = static_cast<double_t>(distance);
        //             return (1.0 / M_PI) * (temperature / (d * d + temperature * temperature));
        //         }
        //     };
        //
        //     struct StableHalf
        //     {
        //         double_t operator()(std::size_t distance, double_t temperature) const
        //         {
        //             double_t d = static_cast<double_t>(distance);
        //             if(d == 0.0)
        //                 return 0.0;
        //             return (1.0 / std::sqrt(2.0 * M_PI * std::pow(d, 3))) * std::exp(-1.0 / (2.0 * d));
        //         }
        //     };
        //} // namespace propabilityFunctions

        //
        //                 using T_propabilityFunction = propabilityFunctions::Exponential;
        //                 static constexpr double T_init = 100.0;
        //                 // magic Number that indicates the lower bound of the
        //                 // temperature used for simulated annealing
        //                 // this should be dependent on the MaxConfigs (more dense search spaces require a lower
        //                 final temperature, yet
        //                 // I need to fit a function here
        //                 static constexpr double T_final = 0.5;
        //
        //                 double_t temperature = T_init; // class member
        //
        //                 double_t calcTemperature(std::size_t maxRuns, std::size_t currentRuns) const
        //                 {
        //                     double_t r = static_cast<double_t>(currentRuns);
        //                     double_t R = static_cast<double_t>(maxRuns);
        //
        //                     if(r >= R)
        //                         return T_final;
        //                     if(r == 0)
        //                         return T_init;
        //
        //                     double_t result = T_init - (T_init - T_final) * std::log(1.0 + r) / std::log(1.0 + R);
        //                     return result;
        //                 }
        //
        //                 // #define SimDebug
        //
        //                 template<typename T_Metric, typename T_ConfigEntry>
        //                 bool acceptWorseSolution(T_ConfigEntry const& cand, T_ConfigEntry const& cur, double
        //                 temperature)
        //                 {
        //                     double delta
        //                         = alpaka::tune::strategy::SimulatedAnnealing::costDifference<T_Metric,
        //                         T_ConfigEntry>{}(cand, cur);
        //
        //                     if(delta <= 0.0)
        //                         return true; // better or equal → accept
        //
        //                     double prob = std::exp(-delta / temperature);
        //         #ifdef SimDebug
        //                     std::cout << "Propability of accepting worse solution is: " << prob << std::endl;
        //         #endif
        //                     std::uniform_real_distribution<double> dist(0.0, 1.0);
        //
        //                     return dist(RNG::get()) < prob;
        //                 }
        //
        //                 template<typename T_Config>
        //                 struct isConfig
        //                 {
        //                     ConfigEntry<T_Config> const& config;
        //                     explicit isConfig(ConfigEntry<T_Config> const& config) : config(config) {};
        //
        //                     template<typename MetricInterface>
        //                     bool betterThan(ConfigEntry<T_Config> const& other)
        //                     {
        //                         auto const& preferred = compareGetBest<MetricInterface>(this->config, other);
        //                         return (&preferred == &this->config);
        //                     }
        //                 };
        //
        //                 /*
        //                  *TODO add genericOperator
        //                  */
        //                 template<typename T_Metric, typename T_Config>
        //                 bool acceptanceFunction(
        //                     ConfigEntry<T_Config> const& neu_,
        //                     ConfigEntry<T_Config> const& cur,
        //                     double temperature)
        //                 {
        //                     if(isConfig{neu_}.template betterThan<T_Metric>(cur))
        //                     {
        //         #ifdef SimDebug
        //                         std::cout << " new Config was better, temperature " << temperature << std::endl;
        //
        //         #endif
        //                         return true;
        //                     }
        //         #ifdef SimDebug
        //                     std::cout << " accepted worse solution to encourage exploration at temp" << temperature
        //                     << std::endl;
        //         #endif
        //                     bool a = acceptWorseSolution<T_Metric>(neu_, cur, temperature);
        //         #ifdef SimDebug
        //                     if(a)
        //                     {
        //                         std::cout << " accepted worse solution to encourage exploration at temp" <<
        //                         temperature
        //                         << std::endl;
        //                     }
        //                     else
        //                     {
        //                         std::cout << " rejected proposed Config as it was slower " << temperature <<
        //                         std::endl;
        //                     }
        //         #endif
        //                     return std::move(a);
        //                 }
        //
        //                 /*
        //                  * accept a already stored ParameterConfiguration with the likelyhood of the acceptance
        //                  function
        //                  */
        //                 template<typename T_MetricInterface, typename T_Config, typename T_KernelRun>
        //                 void acceptNewKernel(
        //                     ConfigEntry<T_Config>& oldKernel,
        //                     ConfigEntry<T_Config>& newKernel,
        //                     KernelTuningModelView<T_KernelRun>& activeKernel,
        //                     double_t temperature)
        //                 {
        //                     if(acceptanceFunction<T_MetricInterface>(newKernel, oldKernel, temperature))
        //                     {
        //         #ifdef SimDebug
        //                         std::cout << " accepted new Kernel" << std::endl;
        //                         activeKernel.fromConfig(newKernel);
        //         #endif
        //                         // toActive(activeKernel, newKernel); -> we dont have to do anything since
        //                         ActiveKernel is already
        //                         // in the newKernel Config
        //                     }
        //                     else
        //                     {
        //         #ifdef SimDebug
        //                         std::cout << " kernel remained" << std::endl;
        //                         activeKernel.fromConfig(oldKernel);
        //         #endif
        //                     }
        //     }
        //
        //     /*
        //      * applys a heuristic on a parameter to select the neighboorhood for each parameter individually based
        //      on a
        //      * propability function that is affected by the cooling rate,
        //      * inspired by:
        //      * https://citeseerx.ist.psu.edu/document?doi=c8dcf69dbc8c750b2db5f16e1e737017efd7dd4a&repid=rep1&type=pdf&utm_source=chatgpt.com
        //      * in the paper they use the hamming distance between two configurations which would include all
        //      parameters
        //      * but applying it per parameter simplifies the computation and algorithm
        //      */
        //     template<typename T_propabilityFunction, typename T>
        //     auto applyProbabilityFunction(std::size_t currentIndex, std::vector<T> const& valueList, double
        //     temperature)
        //
        //     {
        //         using type = std::size_t;
        //
        //         type backwardSteps = currentIndex;
        //         type forwardSteps = valueList.size() - currentIndex - 1;
        //
        //         std::vector<double> weights(backwardSteps + forwardSteps + 1);
        //         double total = 0.0;
        //
        //         // Fill weights: backward (including current) to front
        //         for(type i = 0; i <= backwardSteps; ++i)
        //         {
        //             double weight = T_propabilityFunction{}(i, temperature);
        //             weights[backwardSteps - i] = weight;
        //             total += weight;
        //         }
        //
        //         // Fill weights: forward direction
        //         for(type i = 1; i <= forwardSteps; ++i)
        //         {
        //             double weight = T_propabilityFunction{}(i, temperature);
        //             weights[backwardSteps + i] = weight;
        //             total += weight;
        //         }
        //
        //
        //         // Normalize
        //         for(auto& w : weights)
        //             w /= total;
        //
        //         std::discrete_distribution<type> dist(weights.begin(), weights.end());
        //         type sampledIndex = dist(RNG::get());
        //
        //
        //         // Compute new index
        //         type newIndex;
        //         if(sampledIndex < backwardSteps)
        //         {
        //             type stepsBack = backwardSteps - sampledIndex;
        //             newIndex = currentIndex - stepsBack;
        //         }
        //         else
        //         {
        //             type stepsForward = sampledIndex - backwardSteps;
        //             newIndex = currentIndex + stepsForward;
        //         }
        //
        //         return newIndex;
        //     }
        //
        // #define SimA_MaxCachedSteps 100
        //     std::size_t currentRuns = 0;
        //
        //     template<typename T_Config>
        //     bool isValid(ConfigEntry<T_Config>& entry)
        //     {
        //         return entry.state == ConfigState::Initialized;
        //     }
        //
        //     /// returns true only if both are valid
        //     template<typename T_Config>
        //     bool handleInvalidCases(ConfigEntry<T_Config>& neu, ConfigEntry<T_Config>& old, auto& model)
        //     {
        //         // assumption: model is at the neu Config
        //         if(neu.state == ConfigState::Dummy || old.state == ConfigState::Dummy)
        //         {
        //             if(neu.state == ConfigState::Dummy && old.state != ConfigState::Dummy)
        //             {
        //                 // we revert back to the old state if it was valid while the current is not
        //                 model.fromConfig(old);
        //             }
        //             // this means we automatically accept the "new" state encouraging exploration in case both are
        //             // invalid
        //             return false;
        //         }
        //         if(neu.state == ConfigState::Dummy)
        //         {
        //             return false;
        //         }
        //         if(neu.getMetrics().empty() || old.getMetrics().empty())
        //             return false;
        //         return true;
        //     }
        //
        //     template<typename T_propabilityFunction>
        //     std::size_t sampleHammingDistance(std::size_t maxDistance, double temperature)
        //     {
        //         std::vector<double> weights(maxDistance);
        //         double total = 0.0;
        //
        //         for(std::size_t d = 1; d <= maxDistance; ++d)
        //         {
        //             weights[d - 1] = T_propabilityFunction{}(d, temperature);
        //             total += weights[d - 1];
        //         }
        //
        //         for(auto& w : weights)
        //             w /= total;
        //
        //         std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
        //         return dist(RNG::get()) + 1;
        //     }
        //
        //     template<typename T_Model, typename T_propabilityFunction>
        //     void selectFromNeighborhood(T_Model& model, double temperature)
        //     {
        //         auto& parameters = model.getUniformInterface();
        //         constexpr std::size_t totalParams = std::tuple_size_v<std::decay_t<decltype(parameters)>>;
        //
        // #ifdef SimDebug
        //         std::cout << "[Debug] Total parameters: " << totalParams << "\n";
        // #endif
        //
        //         // Sample a Hamming distance between 1 and totalParams
        //         std::size_t hamming_d = sampleHammingDistance<T_propabilityFunction>(totalParams, temperature);
        //
        // #ifdef SimDebug
        //         std::cout << "[Debug] Sampled Hamming distance: " << hamming_d << "\n";
        // #endif
        //
        //         if(hamming_d > totalParams)
        //         {
        //             throw std::runtime_error("Hamming distance exceeds number of parameters.");
        //         }
        //
        //         // Randomly select `hamming_d` distinct parameter indices
        //         std::bitset<totalParams> selectedFlags;
        //         std::size_t selectedCount = 0;
        //
        //         while(selectedCount < hamming_d)
        //         {
        //             std::size_t idx = std::uniform_int_distribution<std::size_t>{0, totalParams - 1}(RNG::get());
        //             if(!selectedFlags.test(idx))
        //             {
        //                 selectedFlags.set(idx);
        //                 ++selectedCount;
        //
        // #ifdef SimDebug
        //                 std::cout << "[Debug] Selected parameter index for mutation: " << idx << "\n";
        // #endif
        //             }
        //         }
        //
        //         // Apply mutation only to selected indices
        //         std::size_t paramIdx = 0;
        //         for_each(
        //             parameters,
        //             [&](auto& parameter)
        //             {
        //                 if(selectedFlags.test(paramIdx))
        //                 {
        //                     auto oldIndex = parameter.index;
        //                     auto oldValue = parameter.value;
        //
        //                     std::size_t newIndex
        //                         = applyProbabilityFunction(parameter.index, parameter.getValues(), temperature);
        //
        //
        //                     parameter.index = newIndex;
        //                     parameter.value = parameter.getValues()[newIndex];
        //
        // #ifdef SimDebug
        //                     std::cout << "[Debug] Mutated parameter " << paramIdx << ": index " << oldIndex << " → "
        //                               << newIndex << ", value: " << oldValue << " → " << parameter.value << "\n";
        // #endif
        //                 }
        //                 ++paramIdx;
        //             });
        //
        // #ifdef SimDebug
        //         std::cout << "[Debug] Finished parameter mutation.\n";
        // #endif
        //     }
        //
        //     std::optional<std::any> m_lastReturnedKernel;
        //     bool init = true;
        //     uint32_t totalNrOfCacheSteps = 0;
        //     uint32_t nr_times_weChoseOld = 0;
        //     std::optional<std::any> lastConfig;
        //
        //     template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        //     auto operator()(
        //         T_metricInterface&,
        //         KernelTuningModelView<T_TuningModel>& model,
        //         ConfigStorage<T_Config>& config_storage,
        //         EnvironmentState<T_Config>& environmentState)
        //     {
        //         currentRuns = 0;
        //         temperature = calcTemperature(environmentState.getMaxEvals(), environmentState.numValidConfigs);
        //
        //         auto access = lastEvaluatedConfigAccessor<ConfigEntry<T_Config>>();
        //         T_Config oldConfig = model.toConfig();
        //         if(access.has_value())
        //         {
        //             if(m_lastReturnedKernel.has_value())
        //             {
        //                 oldConfig = std::any_cast<T_Config>(m_lastReturnedKernel.value());
        //             }
        //
        //             auto wrapper = access.value();
        //             ConfigEntry<T_Config>& lastEvalEntry = wrapper.get();
        //             ConfigEntry<T_Config>& curEntry = config_storage.getOrCreate(oldConfig);
        //             if(isValid(curEntry))
        //             {
        //                 acceptNewKernel<T_metricInterface>(curEntry, lastEvalEntry, model, temperature);
        //             }
        //             else
        //             {
        //                 model.fromConfig(lastEvalEntry.eonfig);
        //                 oldConfig = lastEvalEntry.eonfig;
        //             }
        //         }
        //
        //
        // #ifdef SimDebug
        //         std::cout << " current Config after initial revert: " << oldConfig.toString() << std::endl;
        //         std::cout << " total nr of cache steps: " << totalNrOfCacheSteps << std::endl;
        // #endif
        //
        //         while(currentRuns < SimA_MaxCachedSteps)
        //         {
        // #ifdef SimDebug
        //             std::cout << "[SimA] Iteration " << currentRuns << " — Old Config: " << oldConfig.toString() <<
        //             std::endl;
        // #endif
        //
        //             selectFromNeighborhood(model, temperature);
        //
        //             T_Config newConfig = model.toConfig();
        //
        // #ifdef SimDebug
        //             std::cout << "[SimA] New Config: " << newConfig.toString() << std::endl;
        // #endif
        //
        //             if(newConfig == oldConfig)
        //             {
        //                 ++currentRuns;
        //                 continue;
        //             }
        //
        //             if(!config_storage.contains(newConfig))
        //             {
        //                 m_lastReturnedKernel.emplace(newConfig); // store a copy
        // #ifdef SimDebug
        //                 std::cout << "[SimA] New Config found — returning." << std::endl;
        // #endif
        //                 return;
        //             }
        //
        // #ifdef SimDebug
        //
        // #endif
        //
        //             ConfigEntry<T_Config>& oldEntry = config_storage.getOrCreate(oldConfig);
        //             ConfigEntry<T_Config>& newEntry = config_storage.getOrCreate(newConfig);
        //
        //             if(!handleInvalidCases(newEntry, oldEntry, model))
        //             {
        // #ifdef SimDebug
        //                 std::cout << "[SimA] Rejected due to invalid metric state." << std::endl;
        // #endif
        //                 ++currentRuns;
        //                 continue;
        //             }
        //
        //             totalNrOfCacheSteps++;
        //
        //             auto oldMedian = oldEntry.getMetrics().get(median_t{}).template as<t_ns>();
        //             auto newMedian = newEntry.getMetrics().get(median_t{}).template as<t_ns>();
        //
        // #ifdef SimDebug
        //             std::cout << "[SimA] Config was actually already evaluated --- whooo." << std::endl;
        //             std::cout << "[SimA] Comparing median: old = " << oldMedian << ", new = " << newMedian <<
        //             std::endl;
        // #endif
        //
        //             acceptNewKernel<T_metricInterface>(oldEntry, newEntry, model, temperature);
        // #ifdef SimDebug
        //             std::cout << "[SimA] Accepted new Config: " << model.toConfig().toString() << "\n" << std::endl;
        // #endif
        //
        //             oldConfig = model.toConfig();
        //             ++currentRuns;
        //         }
        //     }
        //
        //     // if we already have been to that Config we still jump there with the propability function but we go to
        //     // the next Config afterwards
    };
} // namespace alpaka::tune::strategy
#endif // SIMA_OLDSTUB_H
