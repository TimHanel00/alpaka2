//
// Created by tim on 16.03.25.
//
#ifndef STORAGETYPES_H
#define STORAGETYPES_H
#include "alpaka/mem/concepts.hpp"
#include "alpaka/meta/IntegerSequence.hpp"

#include <alpaka/tune/IO/metricContainer.hpp>
#include <alpaka/tune/tunable/kernelTuningModel.hpp>

#include <cmath>

namespace alpaka::tune::config
{
    /// \brief Lifecycle of a configuration during measurement.
    enum class ConfigState
    {
        Uninitialized, ///< New config which has not been seen yet
        Empty, ///< Seen config with no samples taken yet.
        WarmUp, ///< Collecting warm-up samples (not used in stats or write out).
        Initialized, ///< Collecting steady-state samples.
        CICriteriaReached, ///< CI reached
        Retired, ///< fully evaluated configs
        Invalid ///< -due to constraints; metrics cleared.
    };

    /// \brief Ternary outcome for statistical comparison between two configurations used to allow premature
    /// termination of configuration.
    enum class Comparison
    {
        Less, ///< Left-hand side performs better (e.g., lower time).
        Greater, ///< Right-hand side performs better.
        Inconclusive, ///< Not statistically significant (or insufficient data).
        Invalid ///< comparison with an Invalid configuration involved
    };

    /**
     * @brief Runtime record for one concrete configuration.
     *
     * Wraps a fixed configuration (indices) with a rolling set of timing measurements,
     * enabling O(1) access to medians and statistical comparison within a tuning context.
     *
     * @tparam TConfig Fixed-size index configuration type (hashable & comparable).
     */
    template<typename TConfig>
    struct ConfigRecord
    {
        using ConfigType = TConfig;

        /**
         * @brief Default-construct an empty record in Uninitialized state.
         */
        ConfigRecord() = default;

        /**
         * @brief Construct a record for a specific configuration key.
         * @param cfg Configuration to associate with this record.
         */
        explicit ConfigRecord(TConfig const& cfg) : m_config(cfg)
        {
        }

        /**
         * @brief Compare this record to another using the current statistical test.
         * @param other The other configuration record to compare against.
         * @return Comparison result (Less/Greater/Inconclusive/Invalid).
         */
        auto compare(ConfigRecord const& other) const
        {
            return kruskalCompare(*this, other);
        }

        /**
         * @brief Add a timing sample and update state/flags.
         *
         * Transitions Uninitialized → WarmUp → Initialized once the warm-up
         * threshold is reached. Sets @c fullFlag when the metrics container
         * reports a full sample window.
         *
         * @param val Sample value (e.g., time in nanoseconds).
         */
        void pushMetric(double_t val)
        {
            switch(state)
            {
            case ConfigState::Uninitialized:
                throw std::runtime_error("pushing metrics on a new Config is not allowed!");
                break;
            case ConfigState::Empty:
                if(++warm_up_runs >= warmUpThreshold)
                {
                    metrics.clear();
                    metrics.push<10>(val, false);

                    state = ConfigState::Initialized;
                }
                else
                {
                    state = ConfigState::WarmUp;
                    metrics.push<10>(val, false);
                }
                break;
            case ConfigState::WarmUp:

                if(++warm_up_runs >= warmUpThreshold)
                {
                    metrics.clear();
                    metrics.push<10>(val, false);
                    state = ConfigState::Initialized;
                    nr_runs++;
                }
                else
                {
                    metrics.push<10>(val, false);
                }
                break;
            case ConfigState::Initialized:
                if(metrics.push<10>(val, true))
                    state = ConfigState::CICriteriaReached;
                ++nr_runs;
                break;
            case ConfigState::CICriteriaReached:
                metrics.push<10>(val, false);
                ++nr_runs;
                break;

            case ConfigState::Retired:
                this->metrics.history.push_back(val);
                // only append values to history median does not change anymore
                ++nr_runs;
                break;
            case ConfigState::Invalid:
                metrics.clear();
                break;
            }
        }

        /// Current lifecycle state.
        ConfigState state = ConfigState::Uninitialized;

        /**
         * @brief compare equality by value
         * @param other Record to compare.
         * @return True if both represent the same configuration.
         */
        bool operator==(ConfigRecord const& other) const
        {
            return this->toHash() == other.toHash() && this->m_config == other.m_config;
        }

        /**
         * @brief Inequality.
         * @param other Record to compare.
         * @return True if configurations differ.
         */
        bool operator!=(ConfigRecord const& other) const
        {
            return !(*this == other);
        }

        /**
         * @brief Hash of the underlying configuration.
         * @return Hash value.
         */
        auto toHash() const
        {
            return std::hash<ConfigType>{}(m_config);
        }

        /**
         * @brief Access the underlying configuration.
         * @return Const reference to the configuration.
         */
        TConfig const& getConfig() const
        {
            return m_config;
        }

        /**
         * @brief Strict weak ordering by median (ascending).
         * @param entry Right-hand side record.
         * @return True if this median is less than the other's.
         */
        bool operator<(ConfigRecord const& entry) const
        {
            return this->getMedian() < entry.getMedian();
        }

        /**
         * @brief Strict weak ordering by median (descending).
         * @param entry Right-hand side record.
         * @return True if this median is greater than the other's.
         */
        bool operator>(ConfigRecord const& entry) const
        {
            return this->getMedian() > entry.getMedian();
        }

        /**
         * @brief Median of stored samples (nanoseconds) -> O(1) time access.
         * @return Median value as nanoseconds type.
         */
        auto getMedian() const
        {
            return metrics.get(median_t{}).as<t_ns>();
        }

        void clearMeasurements()
        {
            metrics.clear();
        }

        /**
         * @brief Access the metric container.
         * @return Reference to the metric container.
         */
        [[nodiscard]] MetricContainer const& getMeasurements() const
        {
            return metrics;
        }

        /**
         * @brief Number of steady-state runs recorded.
         * @return Count of runs after warm-up.
         */
        [[nodiscard]] std::size_t getRunCount() const
        {
            return nr_runs;
        }

        // Data
        TConfig m_config; ///< Fixed configuration key.
        std::int64_t stamp{0}; ///< Monotonic marker; -1 indicates constraint violation.
        std::size_t nr_runs = 0; ///< Count of steady-state runs.
        std::size_t warm_up_runs = 0; ///< Count of warm-up runs.
        static constexpr std::size_t warmUpThreshold = 1; ///< Warm-up samples before steady-state.

    private:
        MetricContainer metrics; ///< contains statistics for this record.
    };

    /**
     * @brief Non-parametric comparison (Kruskal–Wallis, df=1) between two measured configs.
     *
     * Pools and ranks samples from both records. If H < 3.841 (α=0.05), the result is
     * Inconclusive. Otherwise returns Less/Greater based on medians. Invalid if @p other
     * is marked Invalid.
     *
     * @tparam T_Config Config type.
     * @param current Left-hand configuration record.
     * @param other   Right-hand configuration record.
     * @return Comparison::Less, ::Greater, ::Inconclusive, or ::Invalid.
     */
    template<typename T_Config>
    inline Comparison kruskalCompare(
        alpaka::tune::config::ConfigRecord<T_Config> const& current,
        alpaka::tune::config::ConfigRecord<T_Config> const& other)
    {
        using T_state = decltype(current.state);
        if(other.state == T_state::Invalid)
        {
            return Comparison::Invalid;
        }
        auto const& lhsVals = current.getMeasurements().getAll();
        auto const& rhsVals = other.getMeasurements().getAll();

        if(lhsVals.size() < 3 || rhsVals.size() < 3)
            return Comparison::Inconclusive; // not enough data

        std::vector<std::pair<double_t, int>> combined; // (value, group)
        combined.reserve(lhsVals.size() + rhsVals.size());
        for(auto v : lhsVals)
            combined.emplace_back(v, 0);
        for(auto v : rhsVals)
            combined.emplace_back(v, 1);

        std::sort(combined.begin(), combined.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

        std::vector<double_t> ranks(combined.size());
        for(std::size_t i = 0; i < combined.size(); ++i)
        {
            std::size_t j = i;
            while(j + 1 < combined.size() && combined[j + 1].first == combined[i].first)
                ++j;

            double_t avgRank = static_cast<double_t>(i + j) / 2.0 + 1.0;
            for(std::size_t k = i; k <= j; ++k)
                ranks[k] = avgRank;

            i = j;
        }

        std::size_t n0 = lhsVals.size();
        std::size_t n1 = rhsVals.size();
        std::size_t N = n0 + n1;

        double_t R0 = 0.0, R1 = 0.0;
        for(std::size_t i = 0; i < combined.size(); ++i)
            (combined[i].second == 0 ? R0 : R1) += ranks[i];

        double_t H = (12.0 / (N * (N + 1))) * (R0 * R0 / n0 + R1 * R1 / n1) - 3 * (N + 1);
        constexpr double_t chiSquareCritical = 3.841; // df=1, α=0.05

        if(H < chiSquareCritical)
            return Comparison::Inconclusive;

        double_t lhsMedian = current.getMeasurements().get(median_t{}).template as<t_ns>();
        double_t rhsMedian = other.getMeasurements().get(median_t{}).template as<t_ns>();
        return (lhsMedian < rhsMedian) ? Comparison::Less : Comparison::Greater;
    }
} // namespace alpaka::tune::config

namespace alpaka::tune::IO
{
    /**
     * @brief Runtime history (active window) of all measured configurations.
     *
     * `ActiveHistory` is a session-local database that records each tested configuration
     * and its corresponding measurement record (`ConfigRecord<TConfig>`). It allows:
     *   - O(1) lookup and update by configuration key.
     *   - Sequential replay of configurations in insertion order.
     *
     * This history is **append-only**: once a configuration is added, it remains valid
     * until the end of the tuning session. This guarantees consistency of references
     * across other tuning subsystems (such as `EnvironmentState` or user strategies).
     *
     * @tparam TConfig
     *         Configuration key type (must be hashable and equality comparable).
     *
     * @note
     *  - Configurations are never removed.
     *  - The map and ordered list share ownership of the same `Entry` instances.
     *  - Thread-safety is not guaranteed; synchronization must be done externally if needed.
     */
    template<typename TConfig>
    class ActiveHistory
    {
    public:
        /// @brief Alias for stored entry type (configuration + metrics).
        using Entry = alpaka::tune::config::ConfigRecord<TConfig>;

        /**
         * @brief Insert or lookup an entry (move overload).
         *
         * If the configuration is new, a new entry is created and tracked in insertion order.
         * Otherwise, returns a reference to the existing one.
         *
         * @param config Configuration to insert or retrieve.
         * @return Reference to the stored entry.
         */
        Entry& getOrCreate(TConfig&& config)
        {
            auto [it, inserted] = entries.try_emplace(config, std::move(config));
            if(inserted)
            {
                orderedHistory.emplace_back(std::ref(it->second));
                orderedHistory.back().get().stamp = orderedHistory.size() - 1;
            }
            return it->second;
        }

        /// \brief Lookup an entry by configuration (mutable access).
        /// \param config Configuration key to lookup.
        /// \return Optional reference to the stored entry, or std::nullopt if not found.
        [[nodiscard]] std::optional<std::reference_wrapper<Entry>> getRecord(TConfig const& config) noexcept
        {
            if(auto it = entries.find(config); it != entries.end())
                return std::ref(it->second);
            return std::nullopt;
        }

        /**
         * @brief Insert or lookup an entry (copy overload).
         *
         * Behaves like the move overload but does not modify the input.
         *
         * @param config Configuration to insert or retrieve.
         * @return Reference to the stored entry.
         */
        Entry& getOrCreate(TConfig const& config)
        {
            auto [it, inserted] = entries.try_emplace(config, config);
            if(inserted)
                orderedHistory.emplace_back(std::ref(it->second));
            return it->second;
        }

        /// @brief Number of tracked configurations.
        [[nodiscard]] uint32_t size() const noexcept
        {
            return static_cast<uint32_t>(entries.size());
        }

        /// @brief Unordered map access (fast lookup).
        [[nodiscard]] std::unordered_map<TConfig, Entry>& getAll() noexcept
        {
            return entries;
        }

        /// @brief Const unordered map access.
        [[nodiscard]] std::unordered_map<TConfig, Entry> const& getAll() const noexcept
        {
            return entries;
        }

        /// @brief Access the configurations in insertion order.
        [[nodiscard]] std::vector<std::reference_wrapper<Entry>> const& getOrderedHistory() const noexcept
        {
            return orderedHistory;
        }

        /// @brief Check existence by entry object.
        [[nodiscard]] bool contains(Entry const& config) const noexcept
        {
            return entries.contains(config.m_config);
        }

        /// @brief Check existence by configuration key.
        [[nodiscard]] bool contains(TConfig const& config) const noexcept
        {
            return entries.contains(config);
        }

    private:
        std::unordered_map<TConfig, Entry> entries{};
        std::vector<std::reference_wrapper<Entry>> orderedHistory{};
    };

    /**
     * @brief Metadata snapshot for a specific kernel tuning context.
     *
     * Holds immutable(ish) descriptive information (device/executor/kernel/metric/specifiers)
     * and bookkeeping for the explored configuration space. It aggregates metadata and a configuration descriptor for
     * the active context.
     *
     * @tparam TConfig             Concrete configuration type (flat, indexable).
     * @tparam T_ParameterAccessor Accessor/descriptor that maps configs <-> tunable values.
     */
    template<typename TConfig, typename T_ParameterAccessor>
    struct KernelTuningMetadata
    {
        using TConfig_type = TConfig;

        /// Descriptor for parameters/tunables (types, names, value extraction).
        T_ParameterAccessor descriptor;
        /// Human-readable identifiers for this context.
        std::string device; ///< e.g. "CPU", "NVIDIA V100", etc.
        std::string executor; ///< e.g. mapping/policy name.
        std::string kernel; ///< kernel identifier.
        std::string targetMetric; ///< primary optimization target, e.g. "time".
        std::string kernelArgs; ///< arguments of the kernelBundle
        std::vector<std::string> specifiers; ///< Session/context specifiers/tags.

        /// Construct from a parameter accessor (and an existing Metadata)
        explicit KernelTuningMetadata(T_ParameterAccessor const& accessor, KernelTuningMetadata&& other)
            : descriptor(accessor)
            , device(std::move(other.device))
            , executor(std::move(other.executor))
            , kernel(std::move(other.kernel))
            , targetMetric(std::move(other.targetMetric))
            , kernelArgs(std::move(other.kernelArgs))
            , specifiers(std::move(other.specifiers))
        {
        }

        /// Construct from a parameter accessor (no ownership transfer of other state).
        explicit KernelTuningMetadata(T_ParameterAccessor const& accessor) : descriptor(accessor)
        {
        }

        /// Bookkeeping flags/counters (managed by the tuning flow).
        bool histEvaluated = false; ///< true if historical data was applied/evaluated.
    };

    /**
     * @brief Build a metadata snapshot from a kernel tuning model.
     *
     * Extracts a parameter accessor from the model (using an empty config) and
     * assembles a @ref KernelTuningMetadata with the given identifiers and tags.
     * Wraps a descriptive information for the active context.
     *
     * @tparam KernelTuningModel  Model exposing ConfigDescriptor and getValuesFromConfig().
     * @param  model              Kernel tuning model instance.
     * @param  device             Human-readable device identifier.
     * @param  exec               Executor/mapping identifier.
     * @param  bundle             Kernel/bundle name.
     * @param  sessionSpecs       Session-level specifiers/tags.
     * @param  targetMetric       Primary optimization target (default: "time").
     * @return KernelTuningMetadata<Config, ParameterAccessor> initialized with descriptors and labels.
     */
    template<typename KernelTuningModel, template<class...> class Bundle, typename T_Kernel, typename... T_Args>
    auto createKernelDataFromModel(
        KernelTuningModel& model,
        std::string const& device,
        std::string const& exec,
        Bundle<T_Kernel, T_Args...> const& bundle,
        std::vector<std::string> const& sessionSpecs,
        std::string const& targetMetric = "time")
    {
        using TConfig
            = decltype(alpaka::tune::ConfigDescriptor<std::remove_cvref_t<KernelTuningModel>>::getEmptyConfig());

        auto parameterAccessor = model.getValuesFromConfig(TConfig{});

        auto data = KernelTuningMetadata<TConfig, std::remove_cvref_t<decltype(parameterAccessor)>>{parameterAccessor};
        data.kernel = alpaka::onHost::demangledName<T_Kernel>();
        std::string argTuple = alpaka::onHost::demangledName<typename KernelBundle<T_Kernel, T_Args...>::ArgTuple>();
        argTuple.replace(0, 14, "");
        argTuple.pop_back();
        data.kernelArgs = argTuple;
        data.device = device;
        data.executor = exec;
        data.targetMetric = targetMetric;
        data.specifiers = sessionSpecs;
        return data;
    };
} // namespace alpaka::tune::IO

#endif // STORAGETYPES_H
