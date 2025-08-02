//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include "alpaka/tune/IO/storageTypes.hpp"
#include "alpaka/tune/utils/Random.h"
#include "alpaka/tune/utils/environmentVars.hpp"
#include "alpaka/tune/utils/tupleHelper.h"
#include "bayesianOptimizer.hpp"

#include <alpaka/tune/active/MetricInterface.hpp>

#include <random>
#include <vector>

namespace alpaka::tune::strategy
{


    template<typename T>
    constexpr bool is_signed_type = std::is_signed_v<T>;

    template<typename T>
    auto randomIdx(std::vector<T> const& valueList)
    {
        if(valueList.empty())
        {
            throw std::runtime_error("randomIdx: valueList is empty");
        }

        std::uniform_int_distribution<std::size_t> dis(0, valueList.size() - 1);
        return T{valueList[dis(RNG::get())]};
    }

    template<typename VecT>
    auto randomIdx(std::array<std::vector<typename VecT::type>, VecT::dim()> const& valueLists)
    {
        VecT result;
        for(std::size_t d = 0; d < VecT::dim(); ++d)
        {
            if(valueLists[d].empty())
                throw std::runtime_error("randomIdx: empty valueList in dimension " + std::to_string(d));

            std::uniform_int_distribution<std::size_t> dis(0, valueLists[d].size() - 1);
            result[d] = valueLists[d][dis(RNG::get())];
        }
        return result;
    }

    struct randomSample
    {
        template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        auto operator()(
            T_metricInterface& metricInterface, // the user specified metricInterface
            KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
            ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend config
            EnvironmentState<T_Config>& environmentState) // contains global break criterias
        {
            for_each(
                model.getUniformInterface(),
                [](auto& parameter)
                {
                    auto val = randomIdx(parameter.getValues());
                    parameter.value = val;
                });
        };
    };

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
            double_t operator()(std::size_t distance, double_t temperature) const
            {
                return std::exp(-static_cast<double_t>(distance) / temperature);
            }
        };

        struct Normal
        {
            double_t operator()(std::size_t distance, double_t temperature) const
            {
                double_t d = static_cast<double_t>(distance);
                return (1.0 / std::sqrt(2.0 * M_PI * temperature)) * std::exp(-d * d / (2.0 * temperature));
            }
        };

        struct Cauchy
        {
            double_t operator()(std::size_t distance, double_t temperature) const
            {
                double_t d = static_cast<double_t>(distance);
                return (1.0 / M_PI) * (temperature / (d * d + temperature * temperature));
            }
        };

        struct StableHalf
        {
            double_t operator()(std::size_t distance, double_t temperature) const
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

        template<typename T_Metric, typename T_ConfigEntry>
        bool acceptWorseSolution(T_ConfigEntry const& cand, T_ConfigEntry const& cur, double temperature)
        {
            double delta
                = alpaka::tune::strategy::SimulatedAnnealing::costDifference<T_Metric, T_ConfigEntry>{}(cand, cur);

            if(delta <= 0.0)
                return true; // better or equal → accept

            double prob = std::exp(-delta / temperature);

            static thread_local std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            return dist(rng) < prob;
        }

        template<typename T_Config>
        struct isConfig
        {
            ConfigEntry<T_Config> const& config;
            explicit isConfig(ConfigEntry<T_Config> const& config) : config(config) {};

            template<typename MetricInterface>
            bool betterThan(ConfigEntry<T_Config> const& other)
            {
                auto const& preferred = compareGetBest<MetricInterface>(this->config, other);
                return (&preferred == &this->config);
            }
        };

        /*
         *TODO add genericOperator
         */
        template<typename T_Metric, typename T_Config>
        bool acceptanceFunction(
            ConfigEntry<T_Config> const& neu_,
            ConfigEntry<T_Config> const& cur,
            double temperature)
        {
            if(isConfig{neu_}.template betterThan<T_Metric>(cur))
            {
                return true;
            }
            return acceptWorseSolution<T_Metric>(neu_, cur, temperature);
        }

        /*
         * accept a already stored ParameterConfiguration with the likelyhood of the acceptance function
         */
        template<typename T_MetricInterface, typename T_Config, typename T_KernelRun>
        void acceptNewKernel(
            ConfigEntry<T_Config>& oldKernel,
            ConfigEntry<T_Config>& newKernel,
            KernelTuningModelView<T_KernelRun>& activeKernel,
            double_t temperature)
        {
            if(acceptanceFunction<T_MetricInterface>(newKernel, oldKernel, temperature))
            {
                activeKernel.fromConfig(newKernel);
                // toActive(activeKernel, newKernel); -> we dont have to do anything since ActiveKernel is already
                // in the newKernel config
            }
            else
            {
                activeKernel.fromConfig(oldKernel);
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
        template<typename T>
        auto applyProbabilityFunction(std::size_t currentIndex, std::vector<T> const& valueList, double temperature)
        {
            using type = std::size_t;

            type backwardSteps = currentIndex;
            type forwardSteps = valueList.size() - currentIndex - 1;

            std::vector<double> weights(backwardSteps + forwardSteps + 1);
            double total = 0.0;

            // Fill weights: backward (including current) to front
            for(type i = 0; i <= backwardSteps; ++i)
            {
                weights[backwardSteps - i] = T_propabilityFunction{}(i, temperature);
                total += weights[backwardSteps - i];
            }

            // Fill weights: forward direction
            for(type i = 1; i <= forwardSteps; ++i)
            {
                weights[backwardSteps + i] = T_propabilityFunction{}(i, temperature);
                total += weights[backwardSteps + i];
            }

            // Normalize
            for(auto& w : weights)
                w /= total;

            std::discrete_distribution<type> dist(weights.begin(), weights.end());
            type sampledIndex = dist(RNG::get());

            // Compute new index
            type newIndex;
            if(sampledIndex < backwardSteps)
            {
                type stepsBack = backwardSteps - sampledIndex;
                newIndex = currentIndex - stepsBack;
            }
            else
            {
                type stepsForward = sampledIndex - backwardSteps;
                newIndex = currentIndex + stepsForward;
            }

            return newIndex;
        }

#define SimA_MaxCachedSteps 300
        std::string previousValidKernel = "";
        std::size_t currentRuns = 0;

        /// returns true only if both are valid
        template<typename T_Config>
        bool handleInvalidCases(ConfigEntry<T_Config>& neu, ConfigEntry<T_Config>& old, auto& model)
        {
            // assumption: model is at the neu config
            if(neu.state == ConfigState::Dummy || old.state == ConfigState::Dummy)
            {
                if(neu.state == ConfigState::Dummy && old.state != ConfigState::Dummy)
                {
                    // we revert back to the old state if it was valid while the current is not
                    model.fromConfig(old);
                }
                // this means we automatically accept the "new" state encouraging exploration in case both are
                // invalid
                return false;
            }
            if(neu.state == ConfigState::Dummy)
            {
                return false;
            }
            return true;
        }

        template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        auto operator()(
            T_metricInterface&, // the user specified metricInterface
            KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
            ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend config
            EnvironmentState<T_Config>& environmentState) // contains global break criterias
        {
            currentRuns = 0;
            temperature = calcTemperature(environmentState.getMaxEvals(), environmentState.numberOfCheckedConfigs);
            while(currentRuns < SimA_MaxCachedSteps)
            {
                // Snapshot current configuration
                T_Config oldConfig = model.toConfig();

                // Apply mutation via probability function
                for_each(
                    model.getUniformInterface(),
                    [this](auto& parameter)
                    {
                        auto oldIndex = parameter.index;
                        auto oldValue = parameter.value;

                        parameter.index
                            = applyProbabilityFunction(parameter.index, parameter.getValues(), temperature);
                        parameter.value = parameter.getValues()[parameter.index];
                    });

                // Snapshot new configuration
                T_Config newConfig = model.toConfig();

                if(newConfig == oldConfig)
                {
                    ++currentRuns;
                    continue;
                }

                if(!config_storage.contains(newConfig))
                {
                    return;
                }

                // Retrieve or create entries for metric evaluation
                ConfigEntry<T_Config>& oldEntry = config_storage.getOrCreate(oldConfig);
                ConfigEntry<T_Config>& newEntry = config_storage.getOrCreate(newConfig);


                // Validate metrics/state before accepting
                if(!handleInvalidCases(newEntry, oldEntry, model))
                {
                    ++currentRuns;
                    continue;
                }
                auto oldMedian = oldEntry.getMetrics().get(median_t{}).template as<t_ns>();
                auto newMedian = newEntry.getMetrics().get(median_t{}).template as<t_ns>();

                acceptNewKernel<T_metricInterface>(oldEntry, newEntry, model, temperature);

                ++currentRuns;
            }
        }

        // if we already have been to that config we still jump there with the propability function but we go to
        // the next config afterwards
    };
#if defined(strategy_bayesianOptimization)
    namespace detail
    {
        template<class UI, class Fn>
        constexpr void for_each_param(UI&& ui, Fn&& fn)
        {
            std::apply([&](auto&... p) { (fn(p), ...); }, ui);
        }

        template<class UI>
        [[nodiscard]] constexpr std::size_t param_count(UI&& ui)
        {
            std::size_t c = 0;
            for_each_param(ui, [&](auto&&) { ++c; });
            return c;
        }

        template<class UI>
        [[nodiscard]] Eigen::VectorXd encode(UI&& ui, std::vector<std::size_t> const& card)
        {
            Eigen::VectorXd v(card.size());
            std::size_t i = 0;
            for_each_param(
                ui,
                [&](auto& p)
                {
                    v(i) = static_cast<double>(p.index) / std::max<std::size_t>(1, card[i] - 1);
                    ++i;
                });
            return v;
        }

        template<class UI>
        inline void restore(UI&& ui, std::vector<std::size_t> const& snap)
        {
            std::size_t i = 0;
            for_each_param(
                ui,
                [&](auto& p)
                {
                    p.index = snap[i];
                    p.value = p.getValues()[p.index];
                    ++i;
                });
        }

        // RNG helpers ------------------------------------------------------------------------------
    } // namespace detail

    template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
    void callRandomSearch(
        T_metricInterface& metricInterface, // the user specified metricInterface
        KernelTuningModelView<T_TuningModel>& model,
        ConfigStorage<T_Config>&
            config_storage, // this already returns the config for a specific kernel backend config
        EnvironmentState<T_Config>& environmentState);

    // -----------------------------------------------------------------------------
    // -----------------------------------------------------------------------------
    // bayesianOptimization – refactored
    // -----------------------------------------------------------------------------
    struct bayesianOptimization
    {
        /* ---------- tunables (unchanged) -------------------------------------- */
        std::size_t candidate_pool_size = 1024;
        std::size_t refresh_per_call = 50;

        /* ---------- state (unchanged) ----------------------------------------- */
        bool initialised_ = false;
        std::vector<std::size_t> cardinality_;
        using Vec = Eigen::VectorXd;
        using RowId = std::size_t;

        std::vector<Vec> X_;
        std::vector<double> y_;
        std::unordered_map<std::size_t, RowId> idx_map_;

        GaussianProcess gp_{2.0};
        MultiAcquisition multi_;

        struct Cand
        {
            std::size_t hash;
            Vec vec;
        };

        std::deque<Cand> pool_; // now a deque for cheap pop-front

        /* ---------- configuration -------------------------------------------- */
        static constexpr std::size_t cleanup_interval = 50;
        std::size_t call_counter_ = 0;

        template<typename UI>
        static auto snapshot_indices(UI&& ui) -> std::vector<std::size_t>;
        template<typename UI>
        static Vec encode(UI&& ui, std::vector<std::size_t> const& card);
        template<typename UI>
        static void restore(UI&& ui, std::vector<std::size_t> const& snap);

        template<typename T_Config, typename UI>
        void ingest_or_update(T_Config const& cfg, double median, UI& ui, bool& gp_dirty)
        {
            std::size_t const h = cfg.toHash();
            auto it = idx_map_.find(h);

            if(it == idx_map_.end()) // new config
            {
                Vec v = encode(ui, cardinality_);
                RowId row = X_.size();
                X_.push_back(std::move(v));
                y_.push_back(median);
                idx_map_[h] = row;
                gp_dirty = true;
            }
            else if(y_[it->second] != median) // updated metric
            {
                y_[it->second] = median;
                gp_dirty = true;
            }
        }

        void remove_from_gp(std::size_t h)
        {
            if(auto it = idx_map_.find(h); it != idx_map_.end())
            {
                RowId row = it->second, last = X_.size() - 1;

                if(row != last) // move last row over the gap
                {
                    X_[row] = std::move(X_[last]);
                    y_[row] = y_[last];
                    for(auto& kv : idx_map_) // repair idx that pointed to 'last'
                        if(kv.second == last)
                        {
                            kv.second = row;
                            break;
                        }
                }
                X_.pop_back();
                y_.pop_back();
                idx_map_.erase(it);
            }
        }

        template<typename Metric, typename T_Model, typename T_Config, typename UI_type>
        bool enqueue_random(
            Metric& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env,
            std::vector<std::size_t> const& snap,
            UI_type& ui_ref) // UI_type alias omitted for brevity
        {
            callRandomSearch(metricInterface, model, history, env);
            T_Config cfg = model.toConfig();
            std::size_t h = cfg.toHash();
            if(idx_map_.count(h))
                return false; // already training data

            Vec v = detail::encode(ui_ref, cardinality_);
            pool_.push_back({h, std::move(v)});
            model.fromConfig(cfg); // restore model state
            detail::restore(ui_ref, snap);
            return true;
        }

        template<typename Metric, typename T_Model, typename T_Config, typename UI_type>
        void maintain_pool(
            Metric& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env,
            std::vector<std::size_t> const& snap,
            UI_type& ui_ref,
            std::size_t current_h)
        {
            /* cleanup (every cleanup_interval calls) */
            if(++call_counter_ % cleanup_interval == 0)
            {
                pool_.erase(
                    std::remove_if(
                        pool_.begin(),
                        pool_.end(),
                        [&](Cand const& c) { return history.contains(c) || idx_map_.count(c.hash); }),
                    pool_.end());
            }

            /* initial fill / periodic refresh */
            while(pool_.size() < candidate_pool_size)
            {
                std::size_t added = 0;
                while(added < refresh_per_call && pool_.size() < candidate_pool_size)
                    added += enqueue_random(metricInterface, model, history, env, snap, ui_ref);
                if(added == 0)
                    break; // nothing new could be generated
            }
        }

    public:
        template<concepts::MetricInterface T_metricInterface, typename T_Model, typename T_Config>
        void operator()(
            T_metricInterface& metricInterface,
            KernelTuningModelView<T_Model>& model,
            ConfigStorage<T_Config>& history,
            EnvironmentState<T_Config>& env)
        {
            init_once(model, history); // discover cardinalities

            auto& ui_ref = model.getUniformInterface();
            auto snapshot = snapshot_indices(ui_ref); // save param indices

            /* ingest the freshly-measured config & metric ------------------ */
            T_Config const cfg = model.toConfig();
            std::size_t const h = cfg.toHash();
            bool gp_dirty = false;

            if(history.getOrCreate(cfg).state == ConfigState::Dummy)
            {
                remove_from_gp(h);
                gp_dirty = true;
                randomSample{}(metricInterface, model, history, env); // fallback
                return;
            }

            double median = history.get(cfg).getMetrics().get(median_t{}).template as<t_ns>();
            ingest_or_update(cfg, median, ui_ref, gp_dirty);

            if(gp_dirty && !X_.empty())
                gp_.fit(X_, y_);

            /* maintain candidate pool - the pool of configs from which the suggested config will be selected */
            maintain_pool(metricInterface, model, history, env, snapshot, ui_ref, h);

            if(pool_.empty())
            {
                randomSample{}(metricInterface, model, history, env);
                return;
            }

            /* run acquisition & choose candidate*/
            std::vector<Vec> search;
            search.reserve(pool_.size() + X_.size());
            std::vector<bool> mask;
            mask.reserve(search.capacity());
            std::vector<std::optional<double>> obs;
            obs.reserve(search.capacity());

            for(auto const& c : pool_)
            {
                search.push_back(c.vec);
                mask.push_back(false);
                obs.emplace_back();
            }
            for(RowId i = 0; i < X_.size(); ++i)
            {
                search.push_back(X_[i]);
                mask.push_back(true);
                obs.emplace_back(y_[i]);
            }

            int sel = multi_.suggest(gp_, search, mask, obs);
            if(sel < 0 || static_cast<std::size_t>(sel) >= pool_.size())
                sel = 0;

            /* decode vec back onto UI & schedule evaluation ---------------- */
            detail::restore(ui_ref, snapshot); // reset indices first
            {
                std::size_t i = 0;
                detail::for_each_param(
                    ui_ref,
                    [&](auto& p)
                    {
                        auto const& vals = p.getValues();
                        std::size_t n = vals.size();
                        std::size_t idx = static_cast<std::size_t>(
                            std::round(pool_[sel].vec(i) * std::max<std::size_t>(1, n - 1)));
                        p.index = std::min(idx, n - 1);
                        p.value = vals[p.index];
                        ++i;
                    });
            }

            model.fromConfig(model.toConfig()); // trigger kernel re-instantiation
            history.getOrCreate(model.toConfig()); // ensure entry exists
        }
    };
#endif
    template<std::size_t N>
    alpaka::Vec<std::size_t, N> convertVec(std::vector<std::size_t> const& v)

    {
        alpaka::Vec<std::size_t, N> out;
        for(std::size_t i = 0; i < N; ++i)
            out[i] = v[i];
        return out;
    }

    /**
     *this is a exhaustive search method designed to support asymmetric index ranges and initial values that
     *might not even be on the range (meaning: (value-begin)%stride!=0 && (end-value)%stride!=0)
     *
     * */
    struct exhaustiveSearch
    {
        template<auto N, typename T_Model>
        auto computeValueIndices(KernelTuningModelView<T_Model>& model)
        {
            using VecT = alpaka::Vec<std::size_t, N>;
            VecT idx;

            for_each_enumerate(
                model.getUniformInterface(),
                [&](auto& t, std::size_t i)
                {
                    auto& values = t.getValues();
                    using ValueT = std::decay_t<decltype(values[0])>;
                    if constexpr(std::is_arithmetic_v<ValueT>)
                    {
                        std::cout << std::endl;
                        auto it = std::find(values.begin(), values.end(), t.value[0]);
                        if(it == values.end())
                        {
                            std::cerr << "[ERROR] Value not found in candidate list for Tuneable[" << i << "]\n";
                            // there are for sure edge cases where this failsafe introduces some form of inaccurate
                            // global state but atleast this is caught in the outer scope via a finite loop for
                            // strategy calls

                            it = values.begin();
                        }
                        idx[i] = std::distance(values.begin(), it);
                    }
                    else if constexpr(alpaka::concepts::Vector<std::decay_t<ValueT>>)
                    {
                        auto it = std::find_if(
                            values.begin(),
                            values.end(),
                            [&](auto const& v)
                            {
                                auto cmp = v == t.value;
                                bool match = allTrue(cmp);
                                return match;
                            });

                        if(it == values.end())
                        {
                            std::cerr << "[ERROR] No matching vector found for Tuneable[" << i << "]\n";
                            it = values.begin();
                            // there are for sure edge cases where this failsafe introduces some form of inaccurate
                            // global state but atleast this is caught in the outer scope via a finite loop for
                            // strategy calls
                        }

                        idx[i] = std::distance(values.begin(), it);
                    }
                    else
                    {
                        static_assert(!std::is_same_v<ValueT, ValueT>, "Unsupported value type in exhaustive search.");
                    }
                });
            return idx;
        }

        std::vector<std::size_t> dimsVec;
        std::size_t total = 1;
        std::size_t stateCount = 0;
        bool init = false;
#define ExhaustiveSearchRandomInitialization
#ifdef ExhaustiveSearchRandomInitialization
        static constexpr bool randomInit = true;
#else
        static constexpr bool randomInit = false;
#endif
        template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        auto operator()(
            T_metricInterface& metricInterface, // the user specified metricInterface
            KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
            ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend config
            EnvironmentState<T_Config>& environmentState) // contains global break criterias
        {
            using T_interface = decltype(model.getUniformInterface());
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<T_interface>>;
            using VecT = alpaka::Vec<std::size_t, N>;
            if(!init)
            {
                if constexpr(randomInit)
                {
                    randomSample{}(metricInterface, model, config_storage, environmentState);
                }

                dimsVec.clear();
                total = 1;

                for_each_enumerate(
                    model.getUniformInterface(),
                    [&](auto& t, std::size_t i)
                    {
                        std::size_t sz = t.getValues().size();
                        dimsVec.push_back(sz);
                        total *= sz;
                    });

                VecT idx = computeValueIndices<N>(model);
                VecT dimsVecAsVec = convertVec<N>(dimsVec);
                stateCount = 0;
                init = true;
                return;
            }
            if(stateCount >= total)
            {
                std::cout << " state count " << stateCount << " numChecked " << environmentState.numberOfCheckedConfigs
                          << std::endl;
                std::cout << " total " << total << " stateConfigs " << environmentState.maxConfigsTotal << std::endl;
                std::cout << " for some reason we rached the total state count, therefore exhaustive must be "
                             "wrong somehow "
                          << std::endl;
                environmentState.sessionFinished = true;
                model.fromConfig(environmentState.bestConfig);
                return;
            }
            VecT nd = alpaka::mapToND(convertVec<N>(dimsVec), stateCount++);
            for_each_enumerate(
                model.getUniformInterface(),
                [&](auto& t, std::size_t i)
                {
                    auto& vals = t.getValues();
                    if(nd[i] >= vals.size())
                    {
                        std::abort(); // Stop immediately
                    }
                    t.value = vals[nd[i]];
                });
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
            // example special refinenemt for numBlocks ( in this case doesnt get changed on refinement update
            // cycle)
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

        template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        auto operator()(
            T_metricInterface& metricInterface, // the user specified metricInterface
            KernelTuningModelView<T_TuningModel>& model, // contains tuneables and provides accessors
            ConfigStorage<T_Config>& config_storage, // this is the history for a specific kernel backend config
            EnvironmentState<T_Config>& environmentState) // contains global break criterias
        {
            exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);

            auto config = model.toConfig();
            if(config_storage.contains(config))
            {
                exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);
            }

            if(curIteration < m_numIterations)
            {
                if(config_storage.nrOfConfigs + 2 >= environmentState.maxConfigsTotal)
                {
                    auto& best = config_storage.getOrCreate[environmentState.bestConfig];

                    model.fromConfig(best);

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
                        model.allTuneables());

                    curIteration++;

                    // alpaka::tune::recalculateMaxRuns(kernelRun);
                    config_storage.nrOfConfigs = 0;

                    // Optional debug:
                    // std::cout << "[Refinement] Running second exhaustive search...\n";
                    // exhaustiveSearch<T_Metric>{}(tuneables, kernelRun, kernel_data);
                }
            }
        };
    };

    struct randomSearch
    {
        template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
        auto operator()(
            T_metricInterface& metricInterface, // the user specified metricInterface
            KernelTuningModelView<T_TuningModel>& model,
            ConfigStorage<T_Config>&
                config_storage, // this already returns the config for a specific kernel backend config
            EnvironmentState<T_Config>& environmentState) // contains
        {
            randomSample{}(metricInterface, model, config_storage, environmentState);

            if(config_storage.contains(model.toConfig()))
            {
                exhaustiveSearch{}(metricInterface, model, config_storage, environmentState);
            }
        };
    };

    template<concepts::MetricInterface T_metricInterface, typename T_TuningModel, typename T_Config>
    void callRandomSearch(
        T_metricInterface& metricInterface, // the user specified metricInterface
        KernelTuningModelView<T_TuningModel>& model,
        ConfigStorage<T_Config>&
            config_storage, // this already returns the config for a specific kernel backend config
        EnvironmentState<T_Config>& environmentState)

    {
        randomSearch{}(metricInterface, model, config_storage, environmentState);
        return;
    }
} // namespace alpaka::tune::strategy
#endif // STRATEGY_HPP
