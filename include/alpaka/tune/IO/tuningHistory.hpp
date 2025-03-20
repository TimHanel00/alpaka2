//
// Created by tim on 19.03.25.
//

#ifndef TUNINGHISTORY_H
#define TUNINGHISTORY_H
#include "../../../../toml11/include/toml.hpp"

#include <alpaka/tune/IO/storageTypes.hpp>

namespace alpaka::tune
{
    struct TuningHistory
    {
        using T_floating = std::double_t;


        std::unordered_map<std::string, KernelData> m_tuningHistory;

        TuningHistory() = default;

        template<typename T_DeviceHandle, typename T_Exec, typename T_KernelBundle>
        std::shared_ptr<KernelData> getKernelFromHistory(
            T_DeviceHandle device,
            T_Exec exec,
            T_KernelBundle kernelBundle,
            std::vector<std::string> const& sessionSpecs,
            std::string const& targetMetric = "time")
        {
            std::string lookUpHash = std::string("") + alpaka::core::demangledName(device) + core::demangledName(exec)
                                     + core::demangledName(kernelBundle) + targetMetric
                                     + std::accumulate(sessionSpecs.begin(), sessionSpecs.end(), std::string());
            if(m_tuningHistory.contains(lookUpHash))
            {
                // Return a non-owning shared_ptr by using a no-op deleter.
                return {&m_tuningHistory[lookUpHash], [](KernelData*) { /* no deletion performed */ }};
            }
            return nullptr;
        }

        std::unordered_map<std::uintptr_t, KernelData> kernelEvents;
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

        template<bool loadMultiple = false>
        void loadConfig(std::string const& filename)
        {
            std::cout << filename << std::endl;
            try
            {
                // Parse the TOML file directly from the filename.
                auto config = toml::parse(filename);
                // Retrieve the top-level table.
                auto const& config_table = toml::get<toml::table>(config);

                if(!loadMultiple)
                {
                    std::cout << "[DEBUG] loadMultiple is false, clearing tuning history." << std::endl;
                    m_tuningHistory.clear();
                }

                // Iterate over all top-level key/value pairs.
                for(auto const& [key, value] : config_table)
                {
                    // Skip if the value is not a table.
                    if(!value.is_table())
                        continue;
                    auto const& kernelTable = value.as_table();

                    KernelData kernelData;
                    // Retrieve string values safely.
                    // kernelData.device       = kernelTable.at("device");
                    kernelData.device = find_str(kernelTable, "device");
                    kernelData.executor = find_str(kernelTable, "executor");
                    kernelData.kernel = find_str(kernelTable, "kernel");
                    kernelData.targetMetric = find_str(kernelTable, "targetMetric");

                    // Process the "specifiers" array if it exists.
                    if(kernelTable.contains("specifiers"))
                    {
                        try
                        {
                            toml::array const& specifiers = kernelTable.at("specifiers").as_array();
                            for(auto const& specifier : specifiers)
                            {
                                if(specifier.is_string())
                                {
                                    kernelData.specifiers.push_back(specifier.as_string());
                                }
                            }
                        }
                        catch(std::exception const&)
                        {
                            // Ignore if "specifiers" is not an array
                        }
                    }

                    // Process the "runs" array if it exists.
                    if(kernelTable.contains("runs"))
                    {
                        try
                        {
                            auto const& runs = kernelTable.at("runs").as_array();
                            for(auto const& runValue : runs)
                            {
                                StorageKernelRun run;
                                auto const& runTable = runValue.as_table();
                                run.metric = runTable.at("metric").as_floating();
                                run.nr_runs = runTable.at("nrRuns").as_integer();
                                if(runTable.contains("tuneables"))
                                {
                                    try
                                    {
                                        auto const& tuneables = runTable.at("tuneables").as_table();
                                        for(auto const& [tkey, tval] : tuneables)
                                        {
                                            auto value_tuneAble = tval.as_string();
                                            if(tkey == "gridSize")
                                            {
                                                run.gridSize = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else if(tkey == "blockThreadSize")
                                            {
                                                run.threadBlockSize
                                                    = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else
                                            {
                                                run.tuneables.push_back(
                                                    alpaka::tune::StorageTuneable{tkey, value_tuneAble});
                                            }
                                        }
                                    }
                                    catch(std::exception const& e)
                                    {
                                        std::cout << "exception in passing tuneables " << e.what() << std::endl;
                                        // Ignore if "tuneables" is not an array
                                    }
                                }

                                std::string kernelKey = run.toHash();
                                kernelData.sumOfRuns += run.nr_runs;
                                kernelData.runs[kernelKey] = std::move(run);
                            }
                        }
                        catch(std::exception const& e)
                        {
                            std::cout << e.what() << std::endl;
                            // Ignore if "runs" is not an array
                        }
                    }

                    std::string dataHash = kernelData.toHash();
                    m_tuningHistory[dataHash] = std::move(kernelData);
                }
            }
            catch(std::exception const& e)
            {
                std::cout << "[DEBUG] Failed to load or parse config file (" << filename << "): " << e.what()
                          << std::endl;
                m_tuningHistory.clear();
            }
#ifdef DEBUG
            std::cout << "history size after load: " << m_tuningHistory.size() << std::endl;
            for(auto const& [k, v] : m_tuningHistory)
            {
                std::cout << " runs after history " << v.runs.size() << std::endl;
                for(auto const& [k1, v1] : v.runs)
                {
                    std::cout << " tuneable Size: " << v1.tuneables.size() << std::endl;
                }
            }
#endif
        }

        void storeConfig(std::string const& filename)
        {
            toml::table config;
            std::cout << " tuning Size: " << m_tuningHistory.size() << std::endl;
            for(auto const& [key, kernelData] : m_tuningHistory)
            {
                toml::table kernelTable;
#ifdef DEBUG
                std::cout << "KernelData properties:" << std::endl;
                std::cout << "  - Device: " << kernelData.device << std::endl;
                std::cout << "  - Executor: " << kernelData.executor << std::endl;
                std::cout << "  - Kernel: " << kernelData.kernel << std::endl;
                std::cout << "  - Target Metric: " << kernelData.targetMetric << std::endl;
                std::cout << "Checking kernelData.specifiers size: " << kernelData.specifiers.size() << std::endl;
#endif
                kernelTable.emplace("device", kernelData.device);
                kernelTable.emplace("executor", kernelData.executor);
                kernelTable.emplace("kernel", kernelData.kernel);
                kernelTable.emplace("targetMetric", kernelData.targetMetric);


                toml::array specifiers_array;
                for(auto const& vec : kernelData.specifiers)
                {
                    specifiers_array.emplace_back(vec);
                }
                kernelTable.emplace("specifiers", specifiers_array);

                toml::array runsArray;
#ifdef DEBUG
                std::cout << "Checking kernelData.runs size: " << kernelData.runs.size() << std::endl;
#endif
                for(const auto& run : kernelData.runs)
                {
                    toml::table runTable;
                    runTable.emplace("nrRuns", run.second.nr_runs);
                    runTable.emplace("metric", run.second.metric);
                    toml::table tuneablesTable;
#ifdef DEBUG
                    std::cout << "  - Checking tuneables size: " << run.second.tuneables.size() << std::endl;
#endif
                    for(const auto& tuneable : run.second.tuneables)
                    {
                        tuneablesTable.emplace(tuneable.name, toml::value(tuneable.value));
                    }

                    if(run.second.threadBlockSize != std::nullopt)
                    {
                        tuneablesTable.emplace("blockThreadSize", toml::value(run.second.threadBlockSize->value));
                    }
                    else
                    {
#ifdef DEBUG
                        std::cout << "  - ThreadBlockSize is std::nullopt" << std::endl;
#endif
                    }

                    if(run.second.gridSize != std::nullopt)
                    {
                        tuneablesTable.emplace("gridSize", toml::value(run.second.gridSize->value));
                    }
                    else
                    {
#ifdef DEBUG
                        std::cout << "GridSize is std::nullopt" << std::endl;
#endif
                    }

                    runTable.emplace("tuneables", tuneablesTable);

                    runsArray.emplace_back(runTable);
                }
                kernelTable.emplace("runs", runsArray);
                config.emplace(key, kernelTable);
            }

            std::ofstream file(filename);
            if(!file)
            {
                throw std::runtime_error("Failed to open file for writing: " + filename);
            }
            file << toml::format(toml::value(config));

            file.close();
        }
    };
} // namespace alpaka::tune
#endif // TUNINGHISTORY_H
