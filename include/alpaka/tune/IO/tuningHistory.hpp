//
// Created by tim on 19.03.25.
//

#ifndef TUNINGHISTORY_H
#define TUNINGHISTORY_H
#include "../../../../toml11/include/toml.hpp"

#include <alpaka/tune/IO/storageTypes.hpp>

// #define DEBUG_Hist 0

namespace alpaka::tune
{
    namespace history::detail
    {
        inline auto parseToml(std::string const& file) -> std::optional<toml::basic_value<toml::type_config>>
        {
            try
            {
                std::cout << " successfully assigned valeus" << std::endl;
                return std::make_optional(toml::parse(file));
            }
            catch(std::exception const& e)
            {
                std::cerr << "Failed to parse TOML file '" << file << "': " << e.what() << '\n';
                return std::nullopt; // return empty table or consider throwing further
            }
        }
    } // namespace history::detail

    class TuningHistory
    {
    public:
        using T_parsedToml = decltype(toml::parse(""));

    private:
        std::mutex fileMutex;
        std::optional<T_parsedToml> parsed_toml;
        bool load = false;
        // nr of times load has to be called before a store overwrites the file
        std::atomic<long long> nr_StakeHolders;
        // if true the next time store is called the file is ovewritten
        bool deleteContent = false;

    public:
        explicit TuningHistory(std::string const& file)
            : parsed_toml(history::detail::parseToml(file))
            , nr_StakeHolders(0)
            , filename(file)
        {
        }

        static TuningHistory& get(std::string const& filename)
        {
            static std::unordered_map<std::string, TuningHistory> instances;
            auto [it, inserted] = instances.try_emplace(filename, filename);

            ++it->second.nr_StakeHolders;
            return it->second;
        }

        ~TuningHistory() = default;

        auto find_str(auto const& table, auto const& key)
        {
            std::string s;
            if(table.contains(key))
            {
                s = table.at(key).as_string();
            }
            return s;
        }

        std::string filename;

        template<typename TDescriptor, typename TConfig>
        TConfig configFromNameValueList(
            toml::array const& names,
            toml::array const& values,
            TDescriptor const& descriptor)
        {
            using TupleType = typename TConfig::TupleType;

            std::unordered_map<std::string, std::size_t> nameToIndex;
            for(std::size_t i = 0; i < names.size(); ++i)
                nameToIndex[names[i].as_string()] = i;

            return [&]<typename... DescEntries>(std::tuple<DescEntries...>)
            {
                return TConfig{std::make_tuple(
                    parseVecFromFlatList<typename DescEntries::type>(
                        nameToIndex,
                        names,
                        values,
                        DescEntries{}.name)...)};
            }(typename TDescriptor::entries{});
        }

        static constexpr auto suffixes = Vec{'x', 'y', 'z', 'w'};

        template<typename T_Config, typename T_ConfigDescriptor>
        void parseKernelRuns(KernelData<T_Config, T_ConfigDescriptor>& kernelData, toml::table const& kernelTable)
        {
            if(!kernelTable.contains("runs"))
                return;

            auto const& runs = kernelTable.at("runs").as_array();
            auto const& descriptorEntries = kernelData.descriptor.entries;

            for(auto const& runVal : runs)
            {
                auto const& runTable = runVal.as_table();

                auto const& tuneableVals = runTable.at("tuneableVals").as_array();
                auto const& tuneableNames = runTable.at("tuneableNames").as_array();

                if(tuneableVals.size() != tuneableNames.size())
                    continue;

                typename T_Config::TupleType tuple{};
                std::size_t tuneableIndex = 0;
                // Use your helper for clean access to each descriptor + index
                for_each_enumerate(
                    descriptorEntries,
                    [&]<std::size_t I, typename T0>(T0 const& entry)
                    {
                        using VecType = typename std::remove_cvref_t<T0>::vecType;
                        constexpr std::size_t dim = std::remove_cvref_t<T0>::dimension;
                        VecType parsed{};
                        for(std::size_t k = 0; k < dim; ++k)
                        {
                            if(tuneableIndex >= tuneableNames.size())
                                break;
                            if(tuneableNames[tuneableIndex].as_string() != entry.name + "_" + suffixes[k])
                                continue;
                            if(tuneableVals[tuneableIndex].is_integer())
                                parsed[k]
                                    = static_cast<typename VecType::type>(tuneableVals[tuneableIndex].as_integer());
                            else if(tuneableVals[tuneableIndex].is_floating())
                                parsed[k]
                                    = static_cast<typename VecType::type>(tuneableVals[tuneableIndex].as_floating());
                            else
                            {
                                std::cerr << "[parseKernelRuns] Unsupported TOML type for '" << entry.name << "'\n";
                                return;
                            }
                            tuneableIndex++;
                        }

                        std::get<I>(tuple) = parsed;
                    });
                auto config = Config<decltype(tuple)>{std::move(tuple)};
                // Insert the config into storage
                auto& entry = kernelData.configEntries.getOrCreate(std::move(config));

                // Metrics
                if(runTable.contains("metric"))
                {
                    auto const& metricArray = runTable.at("metric").as_array();
                    for(auto const& val : metricArray)
                        entry.pushMetric(val.as_floating());
                }

                // Run count
                if(runTable.contains("nrRuns"))
                    entry.nr_runs = static_cast<std::size_t>(runTable.at("nrRuns").as_integer());

                // Stamp
                if(runTable.contains("stamp"))
                {
                    long long stamp = static_cast<long long>(runTable.at("stamp").as_integer());
                    kernelData.highestStamp = std::max(kernelData.highestStamp, stamp);
                }

                ++kernelData.nrOfConfigs;
            }

            kernelData.highestStamp += 1;
        }

        template<typename T_Config, typename T_ConfigDescriptor>
        bool matchKernelMetadata(
            KernelData<T_Config, T_ConfigDescriptor> const& kernelData,
            toml::table const& kernelTable)
        {
            auto kernel = find_str(kernelTable, "kernel");
            auto device = find_str(kernelTable, "device");
            auto executor = find_str(kernelTable, "executor");
            auto targetMetric = find_str(kernelTable, "targetMetric");

            return kernel == kernelData.kernel && device == kernelData.device && executor == kernelData.executor
                   && targetMetric == kernelData.targetMetric;
        }

        template<typename T_Config, typename T_ConfigDescriptor>
        void loadConfig(KernelData<T_Config, T_ConfigDescriptor>& kernelData)
        {
            if(!parsed_toml.has_value())
                return;
            if(nr_StakeHolders < 1)
            {
                std::cout << " something went wrong, nr of history initializations does not match load calls "
                             "or called store Config before all load calls"
                          << std::endl;
            }
            --nr_StakeHolders;
            try
            {
                toml::table const& config_table = parsed_toml.value().as_table();
                for(auto const& value : config_table | std::views::values)
                {
                    if(!value.is_table())
                        continue;

                    auto const& kernelTable = value.as_table();

                    // Only process matching kernels
                    if(!matchKernelMetadata(kernelData, kernelTable))
                        continue;

                    // Load optional specifiers
                    if(kernelTable.contains("specifiers"))
                    {
                        try
                        {
                            for(auto const& spec : kernelTable.at("specifiers").as_array())
                            {
                                if(spec.is_string())
                                    kernelData.specifiers.push_back(spec.as_string());
                            }
                        }
                        catch(...)
                        {
                        } // optional, silent fail
                    }

                    // Parse and load all valid runs
                    parseKernelRuns(kernelData, kernelTable);
                }
            }
            catch(std::exception const& e)
            {
                std::cerr << "[loadConfig] Error loading TOML: " << e.what() << std::endl;
            }
        }

        // #define DEBUG_Hist 1


        template<typename T_Config, typename T_ConfigDescriptor>
        void storeConfig(KernelData<T_Config, T_ConfigDescriptor>& kernelData)
        {
            toml::table result;
            toml::table kernelTable;

            // Metadata
            kernelTable["kernel"] = kernelData.kernel;
            kernelTable["device"] = kernelData.device;
            kernelTable["executor"] = kernelData.executor;
            kernelTable["targetMetric"] = kernelData.targetMetric;

            // Specifiers
            toml::array specArray;
            for(auto const& spec : kernelData.specifiers)
                specArray.push_back(spec);
            kernelTable["specifiers"] = std::move(specArray);

            // Runs
            toml::array runs;
            for(auto& [config, entry] : kernelData.configEntries.getAll())
            {
                auto const& tuple = config.getValues();

                toml::array tuneableNames;
                toml::array tuneableVals;

                for_each_enumerate(
                    kernelData.descriptor.entries,
                    [&]<std::size_t I, typename T0>(T0 const& desc)
                    {
                        using VecType = typename std::remove_cvref_t<T0>::vecType;
                        constexpr std::size_t dim = std::remove_cvref_t<T0>::dimension;

                        VecType const& vec = std::get<I>(tuple);
                        for(std::size_t k = 0; k < dim; ++k)
                        {
                            tuneableNames.push_back(desc.name + "_" + std::string(1, suffixes[k]));
                            tuneableVals.push_back(static_cast<double>(vec[k]));
                        }
                    });

                toml::table run;
                run["tuneableNames"] = std::move(tuneableNames);
                run["tuneableVals"] = std::move(tuneableVals);
                run["nrRuns"] = static_cast<std::size_t>(entry.getRunCount());

                // Optional: stamp handling if available in future
                // run["stamp"] = ...

                toml::array metricArray;
                for(auto m : entry.getMetrics().getAll())
                    metricArray.push_back(static_cast<double>(m));
                run["metric"] = std::move(metricArray);

                runs.push_back(std::move(run));
            }

            kernelTable["runs"] = std::move(runs);

            // Use identifiable key for the kernel
            std::string key = kernelData.kernel + "-" + kernelData.device + "-" + kernelData.executor;
            result[key] = std::move(kernelTable);
            {
                std::lock_guard lock(fileMutex);
                std::cout << filename << " filename" << std::endl;
                std::ofstream out(filename, nr_StakeHolders == 0 ? std::ios::trunc : std::ios::app);

                if(!out)
                {
                    std::cerr << "[storeConfig] Failed to open or create file: " << filename << std::endl;
                    return; // or throw
                }
                out << toml::format(toml::value{result});
                nr_StakeHolders = -1;
            }
        }
    };
} // namespace alpaka::tune
#endif // TUNINGHISTORY_H
