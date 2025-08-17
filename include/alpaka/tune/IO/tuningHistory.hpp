//
// Created by tim on 19.03.25.
//

#ifndef TUNINGHISTORY_H
#define TUNINGHISTORY_H

#include <alpaka/tune/active/tuningEnvironment.hpp>

#include <filesystem>
namespace fs = std::filesystem;
#include "../../../../toml11/include/toml.hpp"

#include <alpaka/tune/active/updateMetric.hpp>

// #define DEBUG_Hist 0

namespace alpaka::tune
{
    namespace history::detail
    {
        inline auto parseToml(std::string const& file) -> std::optional<toml::basic_value<toml::type_config>>
        {
            try
            {
                if(fs::exists(file))
                {
                    // File exists, parse safely

                    return std::make_optional(toml::parse(file));
                    // ...rest of loading logic...
                }
                return std::nullopt;
            }
            catch(std::exception const& e)
            {
                std::cerr << "Error passing existing  TOML file '" << file << "': " << e.what() << '\n';
                return std::nullopt; // return empty table or consider throwing further
            }
        }
    } // namespace history::detail
#define BestOnly 1
    class TuningHistory
    {
    public:
        using T_parsedToml = decltype(toml::parse(""));

    private:
        bool bestOnly{BestOnly1};
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

            ++it->second.nr_StakeHolders; // read
            ++it->second.nr_StakeHolders; // write
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

        template<
            typename T_MetricInterface,
            typename T_Config,
            typename T_ConfigDescriptor,
            typename T_EnvironmentState>
        void parseKernelRuns(
            KernelData<T_Config, T_ConfigDescriptor>& kernelData,
            toml::table const& kernelTable,
            T_EnvironmentState& env_state)
        {
            if(!kernelTable.contains("runs"))
                return;
            alpaka::tune::benchmark::phaseAccessor(1);
            auto const& runs = kernelTable.at("runs").as_array();
            auto const& descriptorEntries = kernelData.descriptor.entries;

#ifdef Debug
            std::cout << "[parseKernelRuns] Found " << runs.size() << " run entries.\n";
#endif

            for(auto const& runVal : runs)
            {
                auto const& runTable = runVal.as_table();

                auto const& tuneableVals = runTable.at("tuneableVals").as_array();
                auto const& tuneableNames = runTable.at("tuneableNames").as_array();

#ifdef Debug
                std::cout << "[parseKernelRuns] Tuneable names: " << tuneableNames.size()
                          << ", Tuneable values: " << tuneableVals.size() << "\n";
#endif

                if(tuneableVals.size() != tuneableNames.size())
                {
#ifdef Debug
                    std::cerr << "[parseKernelRuns] Skipping run: name/value count mismatch.\n";
#endif
                    continue;
                }

                typename T_Config::TupleType tuple{};
                std::size_t tuneableIndex = 0;

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

                            auto expectedName = entry.name + "_" + suffixes[k];
                            auto foundName = tuneableNames[tuneableIndex].as_string();

                            if(foundName != expectedName)
                                continue;

#ifdef Debug
                            std::cout << "[parseKernelRuns] Matching tuneable: " << foundName << "\n";
#endif

                            if(tuneableVals[tuneableIndex].is_integer())
                            {
                                parsed[k]
                                    = static_cast<typename VecType::type>(tuneableVals[tuneableIndex].as_integer());
                            }
                            else if(tuneableVals[tuneableIndex].is_floating())
                            {
                                parsed[k]
                                    = static_cast<typename VecType::type>(tuneableVals[tuneableIndex].as_floating());
                            }
                            else
                            {
                                std::cout << "[parseKernelRuns] Unsupported TOML type for '" << entry.name << "'\n";
                                return;
                            }

                            ++tuneableIndex;
                        }

                        std::get<I>(tuple) = parsed;
                    });

                auto config = Config<decltype(tuple)>{std::move(tuple)};
                auto& entry = kernelData.configEntries.getOrCreate(std::move(config));

#ifdef Debug
                std::cout << "[parseKernelRuns] Inserted config:\n" << entry.toString() << "\n";
#endif

                // Metrics
                if(runTable.contains("metric"))
                {
                    auto const& metricArray = runTable.at("metric").as_array();
                    if(metricArray.size() > 0)
                    {
                        entry.state = ConfigState::Initialized;
                    }
                    else
                    {
                        if(runTable.contains("stamp"))
                        {
                            long long stamp = static_cast<long long>(runTable.at("stamp").as_integer());
                            if(stamp == -1)
                            {
                                entry.state = ConfigState::Dummy;
                                entry.stamp = -1;
                                entry.fullFlag = true;
                                entry.nr_runs = std::numeric_limits<decltype(entry.nr_runs)>::max();
                            }
                            else
                            {
                                kernelData.configEntries.remove(entry);
                                continue;
                            }
                        }
                    }
                    for(auto const& val : metricArray)
                    {
                        updateMetrics<T_MetricInterface>(entry, kernelData, env_state, val.as_floating());
                        entry.fullFlag = true;
                    }
                    ++env_state.numberOfCheckedConfigs;
                    ++env_state.numValidConfigs;

#ifdef Debug
                    std::cout << "[parseKernelRuns] Metrics added: " << metricArray.size() << "\n";
#endif
                }

                // Run count
                if(runTable.contains("nrRuns"))
                {
                    entry.nr_runs = static_cast<std::size_t>(runTable.at("nrRuns").as_integer());

#ifdef Debug
                    std::cout << "[parseKernelRuns] Run count: " << entry.nr_runs << "\n";
#endif
                }

                // Stamp
                if(runTable.contains("stamp"))
                {
                    long long stamp = static_cast<long long>(runTable.at("stamp").as_integer());
                    entry.stamp = stamp;
#ifdef Debug
                    std::cout << "Stamp: " << stamp << std::endl;
#endif
                    kernelData.highestStamp = std::max(kernelData.highestStamp, stamp);

#ifdef Debug
                    std::cout << "[parseKernelRuns] Updated stamp: " << stamp << "\n";
#endif
                }

                ++kernelData.nrOfConfigs;
            }

            kernelData.highestStamp += 1;

#ifdef Debug
            std::cout << "[parseKernelRuns] Finished. Total configs: " << kernelData.nrOfConfigs
                      << ", Next stamp: " << kernelData.highestStamp << "\n";
#endif
        }

        template<typename T_Config, typename T_ConfigDescriptor>
        bool matchKernelMetadata(
            KernelData<T_Config, T_ConfigDescriptor> const& kernelData,
            toml::table const& kernelTable)
        {
#ifdef Debug
            std::cout << "[matchKernelMetadata] Called\n";
#endif

            auto kernel = find_str(kernelTable, "kernel");
            auto device = find_str(kernelTable, "device");
            auto executor = find_str(kernelTable, "executor");
            auto targetMetric = find_str(kernelTable, "targetMetric");

#ifdef Debug
            std::cout << "  Parsed fields:\n";
            std::cout << "    kernel       = " << kernel << "\n";
            std::cout << "    device       = " << device << "\n";
            std::cout << "    executor     = " << executor << "\n";
            std::cout << "    targetMetric = " << targetMetric << "\n";
#endif

            bool sessionMatch = true;

            if(kernelTable.contains("sessionSpecifiers"))
            {
                try
                {
                    auto const& sessionArray = kernelTable.at("sessionSpecifiers").as_array();

#ifdef Debug
                    std::cout << "  sessionSpecifiers array found with " << sessionArray.size() << " entries\n";
#endif

                    std::vector<std::string> sessionStrings;
                    for(auto const& val : sessionArray)
                    {
                        if(val.is_string())
                        {
                            sessionStrings.emplace_back(val.as_string());
                        }
                        else
                        {
#ifdef Debug
                            std::cout << "    Skipped non-string sessionSpecifier entry\n";
#endif
                        }
                    }

#ifdef Debug
                    std::cout << "  Extracted sessionSpecifiers from TOML:\n";
                    for(auto const& s : sessionStrings)
                        std::cout << "    " << s << "\n";

                    std::cout << "  kernelData.specifiers:\n";
                    for(auto const& s : kernelData.specifiers)
                        std::cout << "    " << s << "\n";
#endif

                    sessionMatch = std::ranges::equal(sessionStrings, kernelData.specifiers);
                }
                catch(std::exception const& e)
                {
#ifdef Debug
                    std::cerr << "[matchKernelMetadata] Error parsing sessionSpecifiers: " << e.what() << "\n";
#endif
                    sessionMatch = false;
                }
            }

#ifdef Debug
            std::cout << "[matchKernelMetadata] Final comparisons:\n";
            std::cout << "  kernelMatch:       " << (kernel == kernelData.kernel) << "\n";
            std::cout << "  deviceMatch:       " << (device == kernelData.device) << "\n";
            std::cout << "  executorMatch:     " << (executor == kernelData.executor) << "\n";
            std::cout << "  targetMetricMatch: " << (targetMetric == kernelData.targetMetric) << "\n";
            std::cout << "  sessionMatch:      " << sessionMatch << "\n";
#endif

            return kernel == kernelData.kernel && device == kernelData.device && executor == kernelData.executor
                   && targetMetric == kernelData.targetMetric && sessionMatch;
        }

        template<typename T_MetricInterface, typename T_Config, typename T_ConfigDescriptor>
        void loadConfig(KernelData<T_Config, T_ConfigDescriptor>& kernelData, auto& env_state)
        {
            if(!parsed_toml.has_value())
            {
                --nr_StakeHolders;
                return;
            }

            if(nr_StakeHolders < 1)
            {
                std::cerr << " something went wrong, nr of history initializations does not match load calls "
                             "or called store Config before all load calls"
                          << std::endl;
                return;
            }


#ifdef Debug
            std::cout << "[loadConfig] Parsing config with " << nr_StakeHolders << " stakeholders left.\n";
#endif

            try
            {
                toml::table const& config_table = parsed_toml.value().as_table();

#ifdef Debug
                std::cout << "[loadConfig] Loaded TOML table with " << config_table.size() << " entries.\n";
#endif

                for(auto const& value : config_table | std::views::values)
                {
                    if(!value.is_table())
                        continue;

                    auto const& kernelTable = value.as_table();

#ifdef Debug
                    std::cout << "[loadConfig] Found kernel table.\n";
#endif

                    // Only process matching kernels
                    if(!matchKernelMetadata(kernelData, kernelTable))
                    {
#ifdef Debug
                        std::cout << "[loadConfig] Kernel metadata did not match — skipping.\n";
#endif
                        continue;
                    }

#ifdef Debug
                    std::cout << "[loadConfig] Kernel metadata matched — processing.\n";
#endif

                    // Load optional specifiers

                    // Parse and load all valid runs
#ifdef Debug
                    std::cout << "[loadConfig] Parsing kernel runs.\n";
#endif
                    parseKernelRuns<T_MetricInterface>(kernelData, kernelTable, env_state);
                }
            }
            catch(std::exception const& e)
            {
                std::cerr << "[loadConfig] Error loading TOML: " << e.what() << std::endl;
            }
            --nr_StakeHolders; // decrease after parsing
        }

        // #define DEBUG_Hist 1

        template<typename T_Config, typename T_ConfigDescriptor>
        std::string makeKernelKey(KernelData<T_Config, T_ConfigDescriptor> const& kernelData)
        {
            std::string key = kernelData.kernel + "-" + kernelData.device + "-" + kernelData.executor + "-"
                              + kernelData.targetMetric;

            // Add sessionSpecifiers as comma-separated list (optional but precise)
            if(!kernelData.specifiers.empty())
            {
                key += "-[";
                for(std::size_t i = 0; i < kernelData.specifiers.size(); ++i)
                {
                    key += kernelData.specifiers[i];
                    if(i != kernelData.specifiers.size() - 1)
                        key += ",";
                }
                key += "]";
            }

            return key;
        }

        template<typename T_Config, typename T_ConfigDescriptor>
        void storeConfig(KernelData<T_Config, T_ConfigDescriptor>& kernelData)
        {
#ifdef Debug
            std::cout << "[storeConfig] Entering function\n";
#endif

            if(!parsed_toml.has_value())
            {
#ifdef Debug
                std::cout << "[storeConfig] parsed_toml not initialized, emplacing empty table\n";
#endif
                parsed_toml.emplace(toml::table{});
            }

            toml::table& config_table = parsed_toml.value().as_table();
            toml::table kernelTable;

            // --- Metadata
#ifdef Debug
            std::cout << "[storeConfig] Writing metadata\n";
#endif
            kernelTable["kernel"] = kernelData.kernel;
            kernelTable["device"] = kernelData.device;
            kernelTable["executor"] = kernelData.executor;
            kernelTable["targetMetric"] = kernelData.targetMetric;

            // --- Specifiers
#ifdef Debug
            std::cout << "[storeConfig] Writing specifiers\n";
#endif
            toml::array specArray;
            for(auto const& spec : kernelData.specifiers)
            {
#ifdef Debug
                std::cout << "  specifier: " << spec << "\n";
#endif
                specArray.push_back(spec);
            }
            kernelTable["specifiers"] = specArray;
            kernelTable["sessionSpecifiers"] = specArray;

            // --- Runs
#ifdef Debug
            std::cout << "[storeConfig] Writing config entries\n";
#endif
            toml::array runs;
            for(auto& [config, entry] : kernelData.configEntries.getAll())
            {
#ifdef Debug
                std::cout << "  Run entry with " << entry.getRunCount() << " runs, stamp = " << entry.stamp << "\n";
#endif
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
                            std::string name = desc.name + "_" + std::string(1, suffixes[k]);
#ifdef Debug
                            std::cout << "    Tuneable: " << name << " = " << static_cast<double>(vec[k]) << "\n";
#endif
                            tuneableNames.push_back(name);
                            tuneableVals.push_back(static_cast<double>(vec[k]));
                        }
                    });

                toml::table run;
                run["tuneableNames"] = tuneableNames;
                run["tuneableVals"] = tuneableVals;
                run["nrRuns"] = static_cast<std::size_t>(entry.getRunCount());

                run["stamp"] = entry.stamp;
#ifdef Debug
                std::cout << " Entry: " << entry.toString() << " orig stamp: " << entry.stamp
                          << "    has stamp: " << run["stamp"] << "\n";
#endif
                toml::array metricArray;
                for(auto m : entry.getMetrics().getAll())
                {
#ifdef Debug
                    std::cout << "    Metric: " << static_cast<double>(m) << "\n";
#endif
                    metricArray.push_back(static_cast<double>(m));
                }
                run["metric"] = metricArray;

                runs.push_back(std::move(run));
            }

            kernelTable["runs"] = std::move(runs);

            std::string key = makeKernelKey(kernelData);
#ifdef Debug
            std::cout << "[storeConfig] Generated kernel key: " << key << "\n";
            std::cout << "[storeConfig] Removing old entry for key if exists\n";
#endif
            config_table.erase(key); // clear old entry
            config_table[key] = std::move(kernelTable);

            --nr_StakeHolders;
#ifdef Debug
            std::cout << "[storeConfig] nr_StakeHolders = " << nr_StakeHolders << "\n";
#endif

            if(nr_StakeHolders == 0)
            {
#ifdef Debug
                std::cout << "[storeConfig] Writing TOML to file: " << filename << "\n";
#endif
                std::ofstream out(filename, std::ios::trunc);
                if(!out)
                {
#ifdef Debug
                    std::cerr << "[storeConfig] Failed to open file: " << filename << "\n";
#endif
                    return;
                }

                out << toml::format(parsed_toml.value());
                nr_StakeHolders = -1;
#ifdef Debug
                std::cout << "[storeConfig] TOML written successfully\n";
#endif
            }

#ifdef Debug
            std::cout << "[storeConfig] Exiting function\n";
#endif
        }
    };

} // namespace alpaka::tune
#endif // TUNINGHISTORY_H
