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
#include "alpaka/tune/active/activeKernel.hpp"
#include "alpaka/tune/active/tuneable.hpp"

#include <cmath>
#include <numeric>
#include <span>
#include <variant>

inline std::vector<std::string> split(std::string const& s, char delimiter = ',')
{
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string token;
    while(std::getline(ss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

inline std::string remove(std::string const& s, std::string const& removeSeq = "")
{
    std::string ret;
    for(auto h : s)
    {
        bool contained = false;
        for(auto const& elem : removeSeq)
        {
            if(elem == h)
            {
                contained = true;
                break;
            }
        }
        if(!contained)
            ret += h;
    }
    return ret;
}

inline std::vector<std::string> getTokens(std::string const& f)
{
    auto const newS = remove(f, "{}");
    return split(newS, ',');
}

template<typename T>
auto vectorFromString(std::string const& s)
{
    if constexpr(alpaka::isVector_v<T>)
    {
        constexpr auto dim = alpaka::getDim(T{});
        using ElementType = typename T::type;
        auto tokens = getTokens(s);

        if(tokens.size() != dim)
            throw std::runtime_error("Mismatch between vector dimension and number of values");
        auto parse = [](std::string const& tok)
        {
            std::istringstream iss(tok);
            ElementType val;
            if(!(iss >> val))
                throw std::runtime_error("Failed to parse vector component");
            return val;
        };

        return [&]<std::size_t... I>(std::index_sequence<I...>)
        { return T{parse(tokens[I])...}; }(std::make_index_sequence<dim>{});
    }
    throw std::runtime_error("tuneable string does not match any known type");
}

template<typename T_Tune>
inline void tuneableFromString(alpaka::tune::StorageTuneable const& s, T_Tune& k)
{
    if constexpr(std::is_same_v<std::remove_const_t<T_Tune>, alpaka::tune::NoTune>)
    {
        return;
    }
    else
    {
        k.value = vectorFromString<ALPAKA_TYPEOF(k.value)>(s.value);
        k.name = s.name;
    }
}

template<typename T>
std::string convertToString(T const& val)
{
    if constexpr(std::is_same_v<T, std::string>)
    {
        return val;
    }
    else if constexpr(std::is_arithmetic_v<T>)
    {
        return std::to_string(val);
    }
    if constexpr(alpaka::isVector_v<T>)
    {
        return val.toString();
    }

    std::__throw_runtime_error("failed to convert to string - tuneable type not allowed");
}

/*
 *supports conversion from primitive type A -> to T=(string|primitive T)
 */
template<typename T, typename U>
T convertToT(U const& value)
{
    if constexpr(std::is_arithmetic_v<T>)
    {
        return static_cast<T>(value);
    }
    else
    {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}

template<typename T_KernelBundle, typename T_FrameSpec>
struct TuningResult
{
    T_KernelBundle m_kernelBundle;
    T_FrameSpec m_frameSpec;
};

struct t_ns
{
};

struct t_ms
{
};

struct t_s
{
};

struct min_t
{
};

struct max_t
{
};

struct mean_t
{
};

struct median_t
{
};

template<typename T>
struct metricWrapper
{
    T value;

    template<typename Unit>
    [[nodiscard]] double_t as() const
    {
        if constexpr(std::is_same_v<Unit, t_ns>)
            return value;
        else if constexpr(std::is_same_v<Unit, t_ms>)
            return value * 1e6;
        else if constexpr(std::is_same_v<Unit, t_s>)
            return value * 1e9;
        else
            static_assert(!sizeof(Unit), "Unsupported time unit");
    }
};

/*
 * Storage Container for metrics such as timings. gives O(1) access to min,max,median,mean and
 * contains a history to preserve order of observations
 */
class timingsContainer
{
public:
    template<std::size_t stepsUntilCICheck = 10>
    bool push(double_t val)
    {
        history.push_back(val); // to track the order of incoming metrics

        if(val < minVal)
            minVal = val;
        if(val > maxVal)
            maxVal = val;

        meanVal = (meanVal * static_cast<double_t>(count) + val) / (static_cast<double_t>(count) + 1);
        ++count;

        if(lower.empty() || val <= lower.top())
            lower.push(val);
        else
            upper.push(val);

        if(lower.size() > upper.size() + 1)
        {
            upper.push(lower.top());
            lower.pop();
        }
        else if(upper.size() > lower.size())
        {
            lower.push(upper.top());
            upper.pop();
        }
        if(history.size() % stepsUntilCICheck == 0)
        {
            // perform CI (confidence Intervall) check
            return ciWithinTolerance();
        }
        return false;
    }

    [[nodiscard]] std::span<double_t const> getAll() const
    {
        return history;
    }

    std::size_t size() const
    {
        return history.size();
    }

    [[nodiscard]] bool empty() const
    {
        return history.empty();
    }

    [[nodiscard]] metricWrapper<double_t> get(min_t) const
    {
        return {minVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(max_t) const
    {
        return {maxVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(mean_t) const
    {
        return {meanVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(median_t) const
    {
        if(count == 0)
            throw std::runtime_error("No elements");
        if(lower.size() == upper.size())
            return {(lower.top() + upper.top()) / 2.0};
        else
            return {lower.top()};
    }

    /**
     * this is an expensive operation which shouldnt be used too frequently, it rebuilds the priority queues from
     * scratch after pop
     * @param n number of elements to get dropped
     * @return dopped elements as an array
     */
    std::vector<double_t> pop(std::size_t n = 1)
    {
        if(n > history.size())
            throw std::runtime_error("Trying to pop more elements than available");

        std::vector<double_t> popped(history.end() - n, history.end());
        history.resize(history.size() - n);

        rebuildFromHistory(); // keep this as a helper

        return popped; // safe copy, caller owns the data
    }

    /**
     * after every k (where k <=> stepsUntilCICheck) steps we perform a check if the 99% Confidence Intervall
     * (indicating 99% certainty that the true
     *
     * median is contained in that range) deviates less then 5% from the detected/observed median
     *
     * doi: 10.1145/2807591.2807644
     * */
    bool ciWithinTolerance(double_t zscore = 2.576 /*z score for 99% CI */, double_t tolerance = 0.05)
    {
        auto const& all = getAll();
        std::vector<double_t> sorted(all.begin(), all.end());
        std::sort(sorted.begin(), sorted.end());

        std::size_t n = sorted.size();
        if(n < 5)
            return false; // Not enough samples for nonparametric CI

        double_t z = zscore; // for 99% CI
        int lowerIdx = std::max(0, static_cast<int>(std::floor((n - z * std::sqrt(n)) / 2)));
        int upperIdx
            = std::min(static_cast<int>(n - 1), static_cast<int>(std::ceil((1 + (n + z * std::sqrt(n)) / 2))));

        double_t median = get(median_t{}).as<t_ns>();
        double_t ciLow = sorted[lowerIdx];
        double_t ciHigh = sorted[upperIdx];

        double_t ciWidth = ciHigh - ciLow;
        double_t allowedRange = tolerance * median;
        // Check if 99% CI width is within 5% of the median
        return (ciWidth / median) <= tolerance;
    }

private:
    std::priority_queue<double_t> lower; // max-heap to allow O(1) median acces
    std::priority_queue<double_t, std::vector<double_t>, std::greater<>> upper; // min-heap

    double_t meanVal = 0.0;
    size_t count = 0;
    double_t minVal = std::numeric_limits<double_t>::max();
    double_t maxVal = std::numeric_limits<double_t>::lowest();

    std::vector<double_t> history;

    void rebuildFromHistory()
    {
        minVal = std::numeric_limits<double_t>::max();
        maxVal = std::numeric_limits<double_t>::lowest();
        meanVal = 0.0;

        while(!lower.empty())
            lower.pop();
        while(!upper.empty())
            upper.pop();
        for(auto const& val : history)
        {
            if(val < minVal)
                minVal = val;
            if(val > maxVal)
                maxVal = val;

            meanVal = (meanVal * static_cast<double_t>(lower.size() + upper.size()) + val)
                      / (static_cast<double_t>(lower.size() + upper.size() + 1));

            if(lower.empty() || val <= lower.top())
                lower.push(val);
            else
                upper.push(val);

            if(lower.size() > upper.size() + 1)
            {
                upper.push(lower.top());
                lower.pop();
            }
            else if(upper.size() > lower.size())
            {
                lower.push(upper.top());
                upper.pop();
            }
        }
    }
};

template<typename T>
constexpr bool is_stat_type_v
    = std::is_same_v<T, min_t> || std::is_same_v<T, max_t> || std::is_same_v<T, mean_t> || std::is_same_v<T, median_t>;

struct StorageKernelRun;

namespace detail
{
    // Shared helper to perform Kruskal-Wallis comparison
    enum class Comparison;

    Comparison kruskalCompare(StorageKernelRun const& lhs, StorageKernelRun const& rhs, double_t alpha);
} // namespace detail

// storage container of a single Run used for history
struct StorageKernelRun
{
    enum class State
    {
        Uninitialized,
        WarmUp,
        Initialized,
    };
    std::vector<alpaka::tune::StorageTuneable> tuneables;
    std::optional<alpaka::tune::StorageTuneable> numBlocksTune{std::nullopt};
    std::optional<alpaka::tune::StorageTuneable> threadBlockSize{std::nullopt};
    std::optional<alpaka::tune::StorageTuneable> numFramesTune{std::nullopt};
    std::optional<alpaka::tune::StorageTuneable> frameExtentTune{std::nullopt};
    timingsContainer metricContainer;
    std::size_t nr_runs{1};
    State state{State::Uninitialized};

    [[nodiscard]] std::string toHash() const
    {
        std::string m;
        for(auto const& tuneable : tuneables)
        {
            m += tuneable.toHash();
        }
        if(numFramesTune.has_value())
            m += numFramesTune.value().toHash();
        if(frameExtentTune.has_value())
            m += frameExtentTune.value().toHash();
        if(numBlocksTune.has_value())
            m += numBlocksTune.value().toHash();
        if(threadBlockSize.has_value())
            m += threadBlockSize.value().toHash();

        return m;
    }

    [[nodiscard]] std::vector<std::reference_wrapper<alpaka::tune::StorageTuneable const>> view() const
    {
        std::vector<std::reference_wrapper<alpaka::tune::StorageTuneable const>> view;

        view.reserve(tuneables.size());
        for(auto const& t : tuneables)
        {
            view.emplace_back(t);
        }
        if(numBlocksTune)
            view.emplace_back(*numBlocksTune);
        if(threadBlockSize)
            view.emplace_back(*threadBlockSize);
        if(numFramesTune)
            view.emplace_back(*numFramesTune);
        if(frameExtentTune)
            view.emplace_back(*frameExtentTune);

        return view;
    }

    template<typename T>
    std::vector<T> convertToValueArray(std::string const& s)
    {
        std::vector<T> result;
        auto tokens = getTokens(s);
        for(auto const& elem : tokens)
        {
            T resultElem;
            std::istringstream iss(elem);
            if(!(iss >> result))
                throw std::runtime_error(
                    "Conversion from: " + iss.str() + " to given type: " + alpaka::core::Demangled<T>(resultElem)
                    + " failed!");
            result.emplace_back(result);
        }
        return result;
    }

    bool fullFlag = false;

    void pushMetric(double_t const& m)
    {
        constexpr std::size_t k = 10;
        fullFlag = metricContainer.push<k>(m);
    }

    template<typename T, std::enable_if_t<is_stat_type_v<T>, int> = 0>
    metricWrapper<double_t> getMetric()
    {
        return metricContainer.get(T{});
    }

    template<typename T, std::enable_if_t<is_stat_type_v<T>, int> = 0>
    [[nodiscard]] metricWrapper<double_t> getMetric() const
    {
        return metricContainer.get(T{});
    }

    template<typename T>
    std::vector<T> convertToCommonValueArray() const
    {
        std::vector<T> result;
        result.reserve(tuneables.size()); // Reserve space to avoid reallocations

        for(auto const& tune : tuneables)
        {
            auto valueArray = convertToValueArray<T>(tune);
            result.insert(
                result.end(),
                std::make_move_iterator(valueArray.begin()),
                std::make_move_iterator(valueArray.end()));
        }
        if(numBlocksTune.has_value())
        {
            auto valueArray = convertToValueArray<T>(numBlocksTune.value());
            result.insert(
                result.end(),
                std::make_move_iterator(valueArray.begin()),
                std::make_move_iterator(valueArray.end()));
        }
        if(threadBlockSize.has_value())
        {
            auto valueArray = convertToValueArray<T>(threadBlockSize.value());
            result.insert(
                result.end(),
                std::make_move_iterator(valueArray.begin()),
                std::make_move_iterator(valueArray.end()));
        }

        return result;
    }

    auto compare(StorageKernelRun& b)
    {
        return detail::kruskalCompare(*this, b, 0.05);
    }
};

namespace detail
{
    // Shared helper to perform Kruskal-Wallis comparison
    enum class Comparison
    {
        Less,
        Greater,
        Inconclusive
    };

    Comparison kruskalCompare(StorageKernelRun const& lhs, StorageKernelRun const& rhs, double_t alpha = 0.05)
    {
        auto const& lhsVals = lhs.metricContainer.getAll();
        auto const& rhsVals = rhs.metricContainer.getAll();

        if(lhsVals.size() < 5 || rhsVals.size() < 5)
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

        // Chi-square critical value for df = 1, alpha = 0.05 is 3.841
        double_t const chiSquareCritical = 3.841;

        if(H < chiSquareCritical)
            return Comparison::Inconclusive;

        double_t lhsMedian = lhs.metricContainer.get(median_t{}).value;
        double_t rhsMedian = rhs.metricContainer.get(median_t{}).value;

        return (lhsMedian < rhsMedian) ? Comparison::Less : Comparison::Greater;
    }
} // namespace detail

struct KernelData
{
    std::unordered_map<std::string, StorageKernelRun> runs;
    std::string device;
    std::string executor;
    std::string kernel;
    std::string targetMetric;
    std::vector<std::string> specifiers;
    bool exhausted = false;
    std::size_t nrOfConfigs{0};

    std::string toHash()
    {
        std::string concatenatedSpecifier = std::accumulate(specifiers.begin(), specifiers.end(), std::string());
        return device + executor + kernel + targetMetric + concatenatedSpecifier;
    }
};

static KernelData createKernelData(
    std::string const& device,
    std::string const& exec,

    std::string const& bundle,
    std::vector<std::string> const& sessionSpecs,
    std::string const& targetMetric = "time")
{
    KernelData data;
    data.kernel = bundle;
    data.device = device;
    data.executor = exec;
    data.targetMetric = targetMetric;
    data.specifiers = sessionSpecs;
    return data;
};

/*
 * small predefined storageContainer to represent a certain state strategy State of a activeKernelRun
 * (since static variables inside strategies) might violate the constraints implied by the sessionSpecifieres
 */


template<typename Tuple, std::size_t... I>
void updateTuneablesImpl(
    Tuple& tup,
    std::vector<alpaka::tune::StorageTuneable> const& storage,
    std::index_sequence<I...>)
{
    // iterate over tuple elements cast stored string values to corresponding tuple type
    ((std::get<I>(tup).name = storage[I].name,
      std::get<I>(tup).value = convertFromString<decltype(std::get<I>(tup).value)>(storage[I].value)),
     ...);
}

// A free function that updates an the configuration found in a storageKernel
template<
    typename T_numFramesTune,
    typename T_frameExtentTune,
    typename T_numBlocksTune,
    typename T_numThreadsTune,
    typename T_UserDefTuneablesTune>
void toActive(
    ActiveKernelRun<T_numFramesTune, T_frameExtentTune, T_numBlocksTune, T_numThreadsTune, T_UserDefTuneablesTune>&
        active,
    StorageKernelRun const& storeKernel)
{
    // Update gridSize if available.
    if(storeKernel.numFramesTune.has_value())
    {
        tuneableFromString(storeKernel.numFramesTune.value(), active.getNumFramesTune());
    }
    if(storeKernel.frameExtentTune.has_value())
    {
        tuneableFromString(storeKernel.frameExtentTune.value(), active.getFrameExtentTune());
    };
    if(storeKernel.numBlocksTune.has_value())
    {
        tuneableFromString(storeKernel.numBlocksTune.value(), active.getNumBlocksTune());
    }
    if(storeKernel.threadBlockSize.has_value())
    {
        tuneableFromString(storeKernel.threadBlockSize.value(), active.getThreadBlockSizeTune());
    }
    constexpr std::size_t tupleSize = std::tuple_size_v<T_UserDefTuneablesTune>;
    updateTuneablesImpl(active.userDefTuneables, storeKernel.tuneables, std::make_index_sequence<tupleSize>{});

    // Update metric by converting the storage string metric to the active kernel's floating type.
    active.metric = storeKernel.getMetric<median_t>().as<t_ns>();
}

template<typename... T_KernelRunArgs>
StorageKernelRun toStore(ActiveKernelRun<T_KernelRunArgs...>& active)
{
    StorageKernelRun result;
    // Convert gridSize.
    if constexpr(ActiveKernelRun<T_KernelRunArgs...>::hasNumFramesTune())
    {
        result.numFramesTune = alpaka::tune::StorageTuneable{
            active.getNumFramesTune().name,
            convertToString(active.getNumFramesTune().value)};
    }
    if constexpr(ActiveKernelRun<T_KernelRunArgs...>::hasFrameExtentTune())
    {
        result.frameExtentTune = alpaka::tune::StorageTuneable{
            active.getFrameExtentTune().name,
            convertToString(active.getFrameExtentTune().value)};
    }
    if constexpr(ActiveKernelRun<T_KernelRunArgs...>::hasNumBlocksTune())
    {
        result.numBlocksTune = alpaka::tune::StorageTuneable{
            active.getNumBlocksTune().name,
            convertToString(active.getNumBlocksTune().value)};
    }
    if constexpr(ActiveKernelRun<T_KernelRunArgs...>::hasThreadBlockSizeTune())
    {
        result.threadBlockSize = alpaka::tune::StorageTuneable{
            active.getThreadBlockSizeTune().name,
            convertToString(active.getThreadBlockSizeTune().value)};
    }

    // Convert threadBlockSize.


    // Convert the metric.
    if(!std::isnan(active.metric))
    {
        result.pushMetric(active.metric);
    }

    // Convert each tuneable in the tuple
    std::apply(
        [&result](auto const&... tuneable)
        {
            ((result.tuneables.emplace_back(
                 alpaka::tune::StorageTuneable{tuneable.name, convertToString(tuneable.value)})),
             ...);
        },
        active.userDefTuneables);
    return result;
}

#endif // STORAGETYPES_H
