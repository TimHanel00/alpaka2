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
#include "alpaka/tune/tuneable.hpp"

#include <cmath>
inline std::vector<std::string> split(const std::string& s, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}
template<typename T>
T convertFromString(const std::string& s)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        std::istringstream iss(s);
        T val;
        iss >> val;
        return val;
    }
    if constexpr (alpaka::isVector_v<T>)
    {
        constexpr auto dim = alpaka::getDim<T>;
        using ElementType = typename T::type;
        auto tokens = split(s, ',');
        if (tokens.size() != dim)
            throw std::runtime_error("Mismatch between vector dimension and number of values");
        auto parse = [](const std::string& tok) {
            std::istringstream iss(tok);
            ElementType val;
            if (!(iss >> val))
                throw std::runtime_error("Failed to parse vector component");
            return val;
        };

        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return T{parse(tokens[I])...};
        }(std::make_index_sequence<dim>{});
    }
    throw std::runtime_error("tuneable string does not match any known type");
}
template<typename T>
std::string convertToString(const T& val)
{
    if constexpr (std::is_same_v<T, std::string>) {
        return val;
    }
    else if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(val);
    }
    if constexpr (alpaka::isVector_v<T>)
    {
        return val.toString();
    }

    std::__throw_runtime_error("failed to convert to string - tuneable type not allowed");
}
/*
 *supports conversion from primitive type A -> to T=(string|primitive T)
*/
template<typename T, typename U>
T convertToT(const U& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        return static_cast<T>(value);
    } else {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
}
template<typename T_KernelBundle,typename T_FrameSpec>
struct TuningResult
{
    T_KernelBundle m_kernelBundle;
    T_FrameSpec m_frameSpec;
};
    //storage container of a single Run used for history
struct StorageKernelRun
{
    std::vector<alpaka::tune::StorageTuneable> tuneables;
    std::optional<alpaka::tune::StorageTuneable> gridSize{std::nullopt};
    std::optional<alpaka::tune::StorageTuneable> threadBlockSize{std::nullopt};
    std::string metric;
    [[nodiscard]] std::string toHash() const
    {
        std::string m;
        for(auto const &tuneable : tuneables)
        {
            m+=tuneable.toHash();
        }
        if (gridSize.has_value())
            m+=gridSize.value().value;
        if (threadBlockSize.has_value())
            m+=gridSize.value().value;
        return m;
    }
};
//concretely defined run used for tuning (1 per Tuning Session)
    template<typename T_GridSize,typename T_BlockSize,typename T_TuneableType>
struct ActiveKernelRun
    {
        using T_floating =double_t;
        std::optional<T_GridSize> gridSize{std::nullopt};
        std::optional<T_BlockSize> threadBlockSize{std::nullopt};
        T_floating metric{};
        T_TuneableType tuneables;
        constexpr ActiveKernelRun() = default;
        constexpr ActiveKernelRun(std::optional<T_GridSize> const& gridSize, std::optional<T_BlockSize> const& blockSize, const T_TuneableType& args)
            : gridSize(std::move(gridSize))
            , threadBlockSize(std::move(blockSize))
            , metric(std::numeric_limits<T_floating>::quiet_NaN())
            , tuneables(args)
        {
        }

        std::string toHash()
        {
            std::string m;
            std::apply([&m](auto const&... args) {
                ((m += args.toHash()), ...); // fold expression over the comma operator
            }, tuneables);
            if(gridSize.has_value())
                m+=gridSize->toHash();
            if(threadBlockSize.has_value())
                m+=threadBlockSize->toHash();
            return m;
            //return "";
        }
    };
    template<typename Tuple, std::size_t... I>
void updateTuneablesImpl(Tuple& tup, const std::vector<alpaka::tune::StorageTuneable>& storage, std::index_sequence<I...>)
{
    // iterate over tuple elements cast stored string values to corresponding tuple type
    ((std::get<I>(tup).name = storage[I].name,
      std::get<I>(tup).value = convertFromString<decltype(std::get<I>(tup).value)>(storage[I].value)), ...);
}

// A free function that updates an ActiveKernelRun from a StorageKernelRun.
template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
void toActive(
    ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>& active,
    const StorageKernelRun& storeKernel)
{
    // Update gridSize if available.
    if (storeKernel.gridSize.has_value())
    {
        active.gridSize = T_GridSize{
            convertFromString<decltype(active.gridSize->value)>(storeKernel.gridSize->value)
        };
    }
    if (storeKernel.threadBlockSize.has_value())
    {
        active.threadBlockSize = T_BlockSize{
            convertFromString<decltype(active.threadBlockSize->value)>(storeKernel.threadBlockSize->value)
        };
    }
    // Update tuneables: ensure the number of storage tuneables matches the number of elements in the tuple.
    constexpr std::size_t tupleSize = std::tuple_size_v<T_TuneableType>;
    updateTuneablesImpl(active.tuneables, storeKernel.tuneables, std::make_index_sequence<tupleSize>{});

    // Update metric by converting the storage string metric to the active kernel's floating type.
    active.metric = convertFromString<typename ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>::T_floating>(storeKernel.metric);
}
template<typename T_GridSize, typename T_BlockSize, typename T_TuneableType>
StorageKernelRun toStore(ActiveKernelRun<T_GridSize, T_BlockSize, T_TuneableType>& active)
    {
        StorageKernelRun result;
        // Convert gridSize.
        if (active.gridSize.has_value())
        {
            result.gridSize = alpaka::tune::StorageTuneable{
                active.gridSize->name,
                convertToString(active.gridSize->value)
            };
        }
        // Convert threadBlockSize.
        if (active.threadBlockSize.has_value())
        {
            result.threadBlockSize = alpaka::tune::StorageTuneable{
                active.threadBlockSize->name,
                convertToString(active.threadBlockSize->value)
            };
        }
        // Convert the metric.
        if(!std::isnan(active.metric))
        {

            result.metric=convertToString(active.metric);
        }else
        {
            std::__throw_runtime_error("No metric assigned to activeKernel!");
        }


        // Convert each tuneable in the tuple.
        std::apply([&result](auto const&... tuneable) {
            ((result.tuneables.emplace_back(
                 alpaka::tune::StorageTuneable{
                     tuneable.name,
                     convertToString(tuneable.value)
                 }
             )), ...);
        }, active.tuneables);
        return result;
    }

#endif //STORAGETYPES_H
