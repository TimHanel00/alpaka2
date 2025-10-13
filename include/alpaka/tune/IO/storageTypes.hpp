//
// Created by tim on 16.03.25.
//
#ifndef STORAGETYPES_H
#define STORAGETYPES_H
#include "alpaka/mem/concepts.hpp"
#include "alpaka/meta/IntegerSequence.hpp"
#include "alpaka/tune/concepts.hpp"
#include "alpaka/tune/tuneable/kernelTuningModel.hpp"

#include <alpaka/tune/IO/config.hpp>
#include <alpaka/tune/IO/metricContainer.hpp>

#include <assert.h>

#include <cmath>
#include <queue>
#include <span>
#include <variant>


enum class ConfigState
{
    Uninitialized,
    WarmUp,
    Initialized,
    Dummy
};

// forward declarations
enum class Comparison;
template<typename TConfig>
struct ConfigEntry;
template<typename T_Config>
Comparison kruskalCompare(ConfigEntry<T_Config> const& current, ConfigEntry<T_Config> const& other);

// storage container of a single Run used for history
template<typename TConfig>
struct ConfigEntry
{
    using ConfigType = TConfig;
    ConfigEntry() = default;

    explicit ConfigEntry(TConfig const& cfg) : config(cfg)
    {
    }

    std::string toString()
    {
        return config.toString();
    }

    auto compare(ConfigEntry& other)
    {
        return kruskalCompare(*this, other);
    }

    void pushMetric(double_t val)
    {
        switch(state)
        {
        case ConfigState::Uninitialized:
            state = ConfigState::WarmUp;
            ++warm_up_runs;
            nr_runs = 0;
            break;
        case ConfigState::WarmUp:
            if(++warm_up_runs >= warmUpThreshold)
            {
                if(metrics.push<10>(val, fullFlag))
                    fullFlag = true;
                state = ConfigState::Initialized;
                nr_runs++;
            }
            break;
        case ConfigState::Initialized:
            if(metrics.push<10>(val, fullFlag))
                fullFlag = true;
            ++nr_runs;
            break;
        case ConfigState::Dummy:
            metrics.clear();
            break;
        }
    }

    ConfigState state = ConfigState::Uninitialized;

    bool operator==(ConfigEntry const& other) const
    {
        return this->toHash() == config.toHash() && this->config == other.config;
    }

    bool operator!=(ConfigEntry const& other) const
    {
        return !(*this == other);
    }

    auto toHash() const
    {
        return config.toHash();
    }

    TConfig const& getConfig() const
    {
        return config;
    }

    bool operator<(ConfigEntry const& entry) const
    {
        return this->getMedian() < entry.getMedian();
    }

    bool operator>(ConfigEntry const& entry) const
    {
        return this->getMedian() > entry.getMedian();
    }

    auto getMedian() const
    {
        return metrics.get(median_t{}).as<t_ns>();
    }

    MetricContainer& getMetrics()
    {
        return metrics;
    }

    [[nodiscard]] std::size_t getRunCount() const
    {
        return nr_runs;
    }

    TConfig config;
    MetricContainer metrics;
    long long int stamp{0}; // signed to indicate constraint violation with -1
    std::size_t nr_runs = 0;
    std::size_t warm_up_runs = 0;
    bool fullFlag = false;
    static constexpr std::size_t warmUpThreshold = 1;
};

namespace std
{
    template<alpaka::tune::concepts::Integral T, auto N>
    struct hash<ConfigEntry<Config<T, N>>>
    {
        std::size_t operator()(ConfigEntry<Config<T, N>> const& c) const noexcept
        {
            return std::hash{c.config};
        }
    };
} // namespace std
// Shared helper to perform Kruskal-Wallis comparison
enum class Comparison
{
    Less,
    Greater,
    Inconclusive,
    Dummy
};

// Kruskal–Wallis is essentially the non-parametric alternative to one-way ANOVA(analysis of variance). (does not
// assume normality)
/*
 *
 *
 */
template<typename T_Config>
inline Comparison kruskalCompare(ConfigEntry<T_Config>& current, ConfigEntry<T_Config>& other)
{
    using T_state = decltype(current.state);
    if(other.state == T_state::Dummy)
    {
        return Comparison::Dummy;
    }
    auto const& lhsVals = current.getMetrics().getAll();
    auto const& rhsVals = other.getMetrics().getAll();

    if(lhsVals.size() < 3 || rhsVals.size() < 3)
        return Comparison::Inconclusive; // not enough data

    std::vector<std::pair<double_t, int>> combined; // (value, group)
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

        double_t avgRank = (i + j) / 2.0 + 1.0;
        for(std::size_t k = i; k <= j; ++k)
            ranks[k] = avgRank;

        i = j;
    }

    std::size_t n0 = lhsVals.size();
    std::size_t n1 = rhsVals.size();
    std::size_t N = n0 + n1;

    double_t R0 = 0.0, R1 = 0.0;
    for(std::size_t i = 0; i < combined.size(); ++i)
    {
        if(combined[i].second == 0)
            R0 += ranks[i];
        else
            R1 += ranks[i];
    }

    double_t H = (12.0 / (N * (N + 1))) * (R0 * R0 / n0 + R1 * R1 / n1) - 3 * (N + 1);
    constexpr double_t chiSquareCritical = 3.841;
    // Chi-square critical value for df = 1, alpha = 0.05 is 3.841


    if(H < chiSquareCritical)
    {
        std::cout << "[ACTUALLY FAILED the kruskal test: " << current.getMedian() << " " << other.getMedian()
                  << std::endl;
        return Comparison::Inconclusive;
    }
    double_t lhsMedian = current.getMetrics().get(median_t{}).template as<t_ns>();
    double_t rhsMedian = other.getMetrics().get(median_t{}).template as<t_ns>();

    return (lhsMedian < rhsMedian) ? Comparison::Less : Comparison::Greater;
}

template<typename TConfig>
class ConfigStorage
{
public:
    using Entry = ConfigEntry<TConfig>;

    Entry& getOrCreate(TConfig&& config)
    {
        auto [iter, h] = entries.try_emplace(config, config);
        return iter->second;
    }

    Entry& getOrCreate(TConfig const& config)
    {
        auto [iter, h] = entries.try_emplace(config, config);
        return iter->second;
    }

    bool remove(TConfig const& config)
    {
        return entries.erase(config) > 0;
    }

    bool remove(ConfigEntry<TConfig> const& configEntry)
    {
        return entries.erase(configEntry.eonfig) > 0;
    }

    uint32_t size()
    {
        return entries.size();
    }

    std::unordered_map<TConfig, Entry>& getAll()
    {
        return entries;
    }

    bool contains(ConfigEntry<TConfig> const& config) const
    {
        return entries.contains(config.eonfig);
    }

    bool contains(TConfig const& config) const
    {
        return entries.contains(config);
    }

private:
    std::unordered_map<TConfig, Entry> entries{};
};

/*
 * This class is associated with a certain tuning context.
 */
template<typename TConfig, typename T_ParameterAccessor>
struct KernelData
{
    ConfigStorage<TConfig> configEntries;
    //contains metaData of tuneables and types
    T_ParameterAccessor descriptor;
    std::string device;
    std::string executor;
    std::string kernel;
    std::string targetMetric;
    std::vector<std::string> specifiers;
    bool exhausted = false;
    bool histEvaluated = false;
    std::size_t nrOfConfigs{0};
    long long int highestStamp{0};
    std::size_t maxRuns{0};
};

template<typename KernelTuningModel>
auto createKernelDataFromModel(
    KernelTuningModel& model,
    std::string const& device,
    std::string const& exec,
    std::string const& bundle,
    std::vector<std::string> const& sessionSpecs,
    std::string const& targetMetric = "time")
{
    using TConfig = decltype(ConfigDescriptor<std::remove_cvref_t<KernelTuningModel>>::getEmptyConfig());
    using T_ParameterAccessor = decltype(model.getValuesFromConfig(TConfig{}));
    // Step 3: Construct and return KernelData
    KernelData<TConfig, T_ParameterAccessor> data{};
    data.kernel = bundle;
    data.device = device;
    data.executor = exec;
    data.targetMetric = targetMetric;
    data.specifiers = sessionSpecs;
    data.description = model.getValuesFromConfig(TConfig{});
    return data;
}

/*
 * small predefined storageContainer to represent a certain state m_strategy State of a activeKernelRun
 * (since static variables inside strategies) might violate the constraints implied by the sessionSpecifieres
 */

#endif // STORAGETYPES_H
