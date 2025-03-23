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
#include "alpaka/tune/active/tuneable.hpp"

#include <cmath>

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
T convertFromString(std::string const& s)
{
    if constexpr(std::is_arithmetic_v<T>)
    {
        std::istringstream iss(s);
        T val;
        iss >> val;
        return val;
    }
    else if constexpr(alpaka::isVector_v<T>)
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

// storage container of a single Run used for history
struct StorageKernelRun
{
    std::vector<alpaka::tune::StorageTuneable> tuneables;
    std::optional<alpaka::tune::StorageTuneable> gridSize{std::nullopt};
    std::optional<alpaka::tune::StorageTuneable> threadBlockSize{std::nullopt};
    double_t metric;
    std::size_t nr_runs{1};

    [[nodiscard]] std::string toHash() const
    {
        std::string m;
        for(auto const& tuneable : tuneables)
        {
            m += tuneable.toHash();
        }
        if(gridSize.has_value())
            m += gridSize.value().toHash();
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
        if(gridSize.has_value())
        {
            auto valueArray = convertToValueArray<T>(gridSize.value());
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

// concretely defined run for a tuning session stores extracted
template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
struct ActiveKernelRun
{
    using T_floating = double_t;
    std::optional<T_GridSize> gridSize{std::nullopt};
    std::optional<T_BlockSize> threadBlockSize{std::nullopt};
    T_floating metric{};
    T_TuneableType tuneables;
    std::size_t maxRuns{};
    std::size_t maxRunsDefault{1};
    constexpr ActiveKernelRun() = default;

    constexpr ActiveKernelRun(
        std::optional<T_GridSize> const& gridSize,
        std::optional<T_BlockSize> const& blockSize,
        T_TuneableType const& args)
        : gridSize(std::move(gridSize))
        , threadBlockSize(std::move(blockSize))
        , metric(std::numeric_limits<T_floating>::quiet_NaN())
        , tuneables(args)
    {
        for_each(
            tuneables,
            [&](auto& argsT)
            {
                std::size_t maxRunsDefault_old = maxRunsDefault;
                maxRunsDefault *= argsT.numSteps();
                if(maxRunsDefault_old > maxRunsDefault)
                {
                    std::cout << " WARNING: Overflow detected during tuning space calculation, ensure "
                                 "you have a max NumofRuns selected!"
                              << std::endl;
                    maxRunsDefault = UINT64_MAX;
                }
            });
        maxRuns = maxRunsDefault;
    };

    std::string toHash()
    {
        std::string m;
        std::apply(
            [&m](auto const&... args)
            {
                ((m += args.toHash()), ...); // fold expression over the comma operator
            },
            tuneables);
        if(gridSize.has_value())
            m += gridSize->toHash();
        if(threadBlockSize.has_value())
            m += threadBlockSize->toHash();
        return m;
        // return "";
    }

    auto createActiveDummyKernel() const
    {
        // Default-constructed tuple
        T_TuneableType defaultTuneables = createDefaultTuple<T_TuneableType>();

        return std::move(
            ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>(std::nullopt, std::nullopt, defaultTuneables));
    }

private:
    template<typename Tuple, std::size_t... I>
    static Tuple createDefaultTupleImpl(std::index_sequence<I...>)
    {
        return Tuple{std::tuple_element_t<I, Tuple>{}...};
    }

    template<typename Tuple>
    static Tuple createDefaultTuple()
    {
        constexpr std::size_t tuple_size = std::tuple_size_v<Tuple>;
        return createDefaultTupleImpl<Tuple>(std::make_index_sequence<tuple_size>{});
    }
};

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

// A free function that updates an ActiveKernelRun from a StorageKernelRun.
template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
void toActive(ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>& active, StorageKernelRun const& storeKernel)
{
    // Update gridSize if available.
    if(storeKernel.gridSize.has_value())
    {
        active.gridSize = T_GridSize{convertFromString<decltype(active.gridSize->value)>(storeKernel.gridSize->value)};
    }
    if(storeKernel.threadBlockSize.has_value())
    {
        active.threadBlockSize = T_BlockSize{
            convertFromString<decltype(active.threadBlockSize->value)>(storeKernel.threadBlockSize->value)};
    }
    constexpr std::size_t tupleSize = std::tuple_size_v<T_TuneableType>;
    updateTuneablesImpl(active.tuneables, storeKernel.tuneables, std::make_index_sequence<tupleSize>{});

    // Update metric by converting the storage string metric to the active kernel's floating type.
    active.metric = storeKernel.metric;
}

template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
StorageKernelRun toStore(ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>& active)
{
    StorageKernelRun result;
    // Convert gridSize.
    if(active.gridSize.has_value())
    {
        result.gridSize
            = alpaka::tune::StorageTuneable{active.gridSize->name, convertToString(active.gridSize->value)};
    }
    // Convert threadBlockSize.
    if(active.threadBlockSize.has_value())
    {
        result.threadBlockSize = alpaka::tune::StorageTuneable{
            active.threadBlockSize->name,
            convertToString(active.threadBlockSize->value)};
    }
    // Convert the metric.
    if(!std::isnan(active.metric))
    {
        result.metric = active.metric;
    }

    // Convert each tuneable in the tuple.
    std::apply(
        [&result](auto const&... tuneable)
        {
            ((result.tuneables.emplace_back(
                 alpaka::tune::StorageTuneable{tuneable.name, convertToString(tuneable.value)})),
             ...);
        },
        active.tuneables);
    return result;
}

#endif // STORAGETYPES_H
