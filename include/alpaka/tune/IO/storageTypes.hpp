//
// Created by tim on 16.03.25.
//

#ifndef STORAGETYPES_H
//
// Created by tim on 16.03.25.
//
#define STORAGETYPES_H
#include "alpaka/core/RemoveRestrict.hpp"
#include "alpaka/meta/IntegerSequence.hpp"
#include "alpaka/tune/active/kernelTuningModel.hpp"

#include <alpaka/tune/IO/metricContainer.hpp>

#include <cmath>
#include <numeric>
#include <queue>
#include <span>
#include <variant>

template<typename T_ConfigTuple>
struct Config

{
    using TupleType = T_ConfigTuple;
    Config() = default;

    explicit Config(T_ConfigTuple&& vals) : values(std::forward<T_ConfigTuple>(vals))
    {
        this->hashVal = computeHash(this->values);
    }

    explicit Config(T_ConfigTuple const& vals) : values(vals)
    {
        this->hashVal = computeHash(this->values);
    }

    std::string toString() const
    {
        return std::apply([](auto const&... args) { return (std::string{} + ... + args.toString()); }, values);
    };

    std::size_t toHash() const
    {
        return hashVal;
    }

    TupleType const& getValues() const
    {
        return values;
    }

    bool operator==(Config const& other) const
    {
        return this->toHash() == other.toHash() && this->values == other.values;
    }

    bool operator!=(Config const& other) const
    {
        return !(*this == other);
    }

    template<typename KModel>
    static Config fromModel(KModel const& model)
    {
        auto tuple = std::apply(
            [&](auto const&... frameElems)
            {
                return std::apply(
                    [&](auto const&... userElems)
                    {
                        return std::apply(
                            [&](auto const&... compileElems)
                            {
                                return std::make_tuple(userElems.value..., frameElems.value..., compileElems.value...);
                                // here we actually copy by value
                            },
                            model.m_compileTimeTuneables);
                    },
                    model.m_userTuneables);
            },
            model.m_frameTuneables);
        return Config{tuple};
    }

private:
    TupleType values;
    std::size_t hashVal;

    static std::size_t computeHash(TupleType const& vals)
    {
        return std::apply(
            [](auto const&... val)
            {
                std::size_t seed = 0;
                (..., (seed ^= hashVec(val) + 0x9e37'79b9 + (seed << 6) + (seed >> 2)));
                return seed;
            },
            vals);
    }

    template<typename Vec>
    static std::size_t hashVec(Vec const& vec)
    {
        std::size_t hash = 0;
        constexpr auto dim = alpaka::getDim(Vec{});
        for(std::size_t i = 0; i < dim; ++i)
        {
            hash ^= std::hash<typename Vec::type>{}(vec[i]) + 0x9e37'79b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};
enum class ConfigState
{
    Uninitialized,
    WarmUp,
    Initialized,
    Dummy
};

// Shared helper to perform Kruskal-Wallis comparison
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
        if(metrics.push<10>(val))
            fullFlag = true;

        switch(state)
        {
        case ConfigState::Uninitialized:
            state = ConfigState::WarmUp;
            ++warm_up_runs;
            break;
        case ConfigState::WarmUp:
            if(++warm_up_runs > warmUpThreshold)
            {
                metrics.clear();
                metrics.push<10>(val);
                state = ConfigState::Initialized;
                nr_runs = 1;
            }
            break;
        case ConfigState::Initialized:
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

    std::size_t getRunCount() const
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

// Shared helper to perform Kruskal-Wallis comparison
enum class Comparison
{
    Less,
    Greater,
    Inconclusive,
    Dummy
};

// Kruskal–Wallis is essentially the non-parametric alternative to one-way ANOVA. (does not assume normality)
template<typename T_Config>
inline Comparison kruskalCompare(ConfigEntry<T_Config>& current, ConfigEntry<T_Config>& other)
{
    using T_state = ALPAKA_TYPEOF(current.state);
    if(other.state == T_state::Dummy)
    {
        return Comparison::Dummy;
    }
    auto const& lhsVals = current.getMetrics().getAll();
    auto const& rhsVals = other.getMetrics().getAll();

    if(lhsVals.size() < 1 || rhsVals.size() < 1)
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
        return Comparison::Inconclusive;
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

    std::unordered_map<TConfig, Entry>& getAll()
    {
        return entries;
    }

    bool contains(ConfigEntry<TConfig> const& config) const
    {
        return entries.contains(config.config);
    }

    bool contains(TConfig const& config) const
    {
        return entries.contains(config);
    }

private:
    std::unordered_map<TConfig, Entry> entries;
};

namespace std
{
    template<typename... Ts>
    struct hash<Config<Ts...>>
    {
        std::size_t operator()(Config<Ts...> const& c) const
        {
            auto hash = c.toHash();
            return hash;
        }
    };
} // namespace std

/*
 * This class is associated with a certain tuning context.
 */
template<typename TConfig, typename T_ConfigDescriptor>
struct KernelData
{
    ConfigStorage<TConfig> configEntries;
    T_ConfigDescriptor descriptor;
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

template<typename T_Vec>
requires(alpaka::isVector_v<T_Vec>)
struct ConfigDescriptorEntry
{
    std::string name;
    ConfigDescriptorEntry() = default;
    static constexpr std::size_t dimension = alpaka::getDim(T_Vec{});
    using type = typename T_Vec::type;
    using vecType = T_Vec;

    explicit ConfigDescriptorEntry(std::string name_) : name(std::move(name_))
    {
    }
};

template<typename... TVec>
struct ConfigDescriptor
{
    using Entries = std::tuple<ConfigDescriptorEntry<TVec>...>;
    Entries entries;
    ConfigDescriptor() = default;

    explicit ConfigDescriptor(std::tuple<ConfigDescriptorEntry<TVec>...> v) : entries(std::move(v))
    {
    }

    static constexpr std::size_t size = sizeof...(TVec);
};
template<typename Tuple>
struct DescriptorFromAllTuneables;

template<typename... Tuneables>
struct DescriptorFromAllTuneables<std::tuple<Tuneables...>>
{
    using type = ConfigDescriptor<ConfigDescriptorEntry<typename std::remove_cvref_t<Tuneables>::ValueType>...>;
};

template<typename Tuple>
auto buildDescriptorFromTuneables(Tuple&& tuneables)
{
    return std::apply(
        [](auto const&... tune)
        {
            return ConfigDescriptor<typename std::remove_cvref_t<decltype(tune)>::ValueType...>(std::make_tuple(
                ConfigDescriptorEntry<typename std::remove_cvref_t<decltype(tune)>::ValueType>{tune.name()}...));
        },
        tuneables);
}

template<typename KernelTuningModel>
auto createKernelDataFromModel(
    KernelTuningModel& model,
    std::string const& device,
    std::string const& exec,
    std::string const& bundle,
    std::vector<std::string> const& sessionSpecs,
    std::string const& targetMetric = "time")
{
    // Step 1: Flatten allTuneables
    auto all = model.allTuneables(); // tuple<Tuneable<T_Vec, ...>...>

    // Step 2: Build Config<Ts...> and Descriptor<...> from tuneables
    using TConfig = decltype(Config{model.allValues()});
    using TupleOfVecs = decltype(std::apply(
        [](auto const&... t) { return std::tuple<typename std::remove_cvref_t<decltype(t)>::ValueType...>{}; },
        all));


    using TDescriptor = decltype(buildDescriptorFromTuneables(all));
    // Step 3: Construct and return KernelData
    KernelData<TConfig, TDescriptor> data{};
    data.kernel = bundle;
    data.device = device;
    data.executor = exec;
    data.targetMetric = targetMetric;
    data.specifiers = sessionSpecs;
    data.descriptor = buildDescriptorFromTuneables(all);
    data.configEntries.getOrCreate(Config{model.allValues()});
    return data;
}

/*
 * small predefined storageContainer to represent a certain state m_strategy State of a activeKernelRun
 * (since static variables inside strategies) might violate the constraints implied by the sessionSpecifieres
 */

#endif // STORAGETYPES_H
