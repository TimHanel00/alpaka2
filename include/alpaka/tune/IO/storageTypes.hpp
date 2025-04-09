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

inline double_t accessMetric(std::priority_queue<double_t, std::vector<double_t>, std::greater<>> const& h)
{
    return h.top();
}

template<typename T_KernelBundle, typename T_FrameSpec>
struct TuningResult
{
    T_KernelBundle m_kernelBundle;
    T_FrameSpec m_frameSpec;
};

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
    std::priority_queue<double_t, std::vector<double_t>, std::greater<>> metric;
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
};

struct KernelData
{
    std::unordered_map<std::string, StorageKernelRun> runs;
    std::string device;
    std::string executor;
    std::string kernel;
    std::string targetMetric;
    std::vector<std::string> specifiers;
    bool exhausted = false;
    std::size_t sumOfRuns{0};

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
        tuneableFromString(storeKernel.numBlocksTune.value(), active.getNumFramesTune());
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
    active.metric = storeKernel.metric.top();
}

template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
StorageKernelRun toStore(ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>& active)
{
    StorageKernelRun result;
    // Convert gridSize.
    if constexpr(active.hasNumFramesTune())
    {
        result.numFramesTune = alpaka::tune::StorageTuneable{
            active.getNumFramesTune().name,
            convertToString(active.getNumFramesTune().value)};
    }
    if constexpr(active.hasFrameExtentTune())
    {
        result.frameExtentTune = alpaka::tune::StorageTuneable{
            active.getFrameExtentTune().name,
            convertToString(active.getFrameExtentTune().value)};
    }
    if constexpr(active.hasNumBlocksTune())
    {
        result.numBlocksTune = alpaka::tune::StorageTuneable{
            active.getNumBlocksTune().name,
            convertToString(active.getNumBlocksTune().value)};
    }
    if constexpr(active.hasThreadBlockSizeTune())
    {
        result.threadBlockSize = alpaka::tune::StorageTuneable{
            active.getThreadBlockSizeTune().name,
            convertToString(active.getThreadBlockSizeTune().value)};
    }

    // Convert threadBlockSize.


    // Convert the metric.
    if(!std::isnan(active.metric))
    {
        result.metric.push(active.metric);
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
