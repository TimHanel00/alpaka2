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
    struct TuningHistory
    {
        using T_floating = std::double_t;

        std::unordered_map<std::string, KernelData> m_tuningHistory;

        static TuningHistory& get(std::string const& config)
        {
            static std::unordered_map<std::string, TuningHistory> instances;
            return instances[config];
        }

        bool initialized = false;
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
                                     + typeid(kernelBundle).name() + targetMetric
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

        template<bool loadMultiple = true>
        void loadConfig(std::string const& filename)
        {
            std::cout << filename << std::endl;
            try
            {
                auto config = toml::parse(filename);
                auto const& config_table = toml::get<toml::table>(config);

                if(!loadMultiple)
                {
                    // std::cout << "[DEBUG] loadMultiple is false, clearing tuning history." << std::endl;
                    m_tuningHistory.clear();
                }

                for(auto const& [key, value] : config_table)
                {
                    if(!value.is_table())
                        continue;

                    auto const& kernelTable = value.as_table();
                    KernelData kernelData;

                    kernelData.device = find_str(kernelTable, "device");
                    kernelData.executor = find_str(kernelTable, "executor");
                    kernelData.kernel = find_str(kernelTable, "kernel");
                    kernelData.targetMetric = find_str(kernelTable, "targetMetric");

#ifdef DEBUG_Hist
                    std::cout << "[DEBUG_Hist] Kernel Key: " << key << std::endl;
                    std::cout << "  - Device: " << kernelData.device << std::endl;
                    std::cout << "  - Executor: " << kernelData.executor << std::endl;
                    std::cout << "  - Kernel: " << kernelData.kernel << std::endl;
                    std::cout << "  - Target Metric: " << kernelData.targetMetric << std::endl;
#endif

                    long long int highestStamp = 0;

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
#ifdef DEBUG_Hist
                            std::cout << "  - Failed to load specifiers." << std::endl;
#endif
                        }
                    }

                    if(kernelTable.contains("runs"))
                    {
                        try
                        {
                            auto const& runs = kernelTable.at("runs").as_array();
                            for(auto const& runValue : runs)
                            {
                                StorageKernelRun run;
                                auto const& runTable = runValue.as_table();

                                if(!runTable.at("metric").as_array().empty())
                                {
                                    run.state = StorageKernelRun::State::Initialized;
                                }
                                if(runTable.at("stamp").as_integer() == -1)
                                {
                                    run.state = StorageKernelRun::State::Dummy;
                                    run.stamp = -1;
                                    run.fullFlag = true;
                                }
                                if(run.state != StorageKernelRun::State::Initialized
                                   && run.state != StorageKernelRun::State::Dummy)
                                    continue;

                                for(auto const& elem : runTable.at("metric").as_array())
                                {
                                    run.pushMetric(elem.as_floating());
                                }


                                run.nr_runs = runTable.at("nrRuns").as_integer();
                                if(runTable.contains("tuneableNames"))
                                {
                                    try
                                    {
                                        run.stamp = static_cast<long long int>(runTable.at("stamp").as_integer());
                                        highestStamp = std::max(highestStamp, run.stamp);

                                        auto const& tuneablesV = runTable.at("tuneableVals").as_array();
                                        auto const& tuneablesID = runTable.at("tuneableNames").as_array();

                                        for(int i = 0; i < tuneablesID.size(); i++)
                                        {
                                            auto tkey = tuneablesID[i].as_string();
                                            auto value_tuneAble = tuneablesV[i].as_string();

                                            if(std::string_view(tkey) == getNameFromTag<frameTune::numBlocks>())
                                            {
                                                run.numBlocksTune
                                                    = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else if(std::string_view(tkey) == getNameFromTag<frameTune::ThreadBlock>())
                                            {
                                                run.threadBlockSize
                                                    = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else if(std::string_view(tkey) == getNameFromTag<frameTune::NumFrames>())
                                            {
                                                run.numFramesTune
                                                    = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else if(std::string_view(tkey) == getNameFromTag<frameTune::FrameExtent>())
                                            {
                                                run.frameExtentTune
                                                    = alpaka::tune::StorageTuneable{tkey, value_tuneAble};
                                            }
                                            else if(std::string_view(tkey).find("CTune") != std::string_view::npos)
                                            {
                                                run.Ctuneables.push_back(
                                                    alpaka::tune::StorageTuneable{tkey, value_tuneAble});
                                            }
                                            else
                                            {
                                                run.tuneables.push_back(
                                                    alpaka::tune::StorageTuneable{tkey, value_tuneAble});
                                            }
                                        }
#ifdef DEBUG_Hist
                                        std::cout << "    - Parsed tuneables: " << run.tuneables.size()
                                                  << ", CTuneables: " << run.Ctuneables.size() << std::endl;
#endif
                                    }
                                    catch(std::exception const& e)
                                    {
                                        std::cout << "exception in parsing tuneables: " << e.what() << std::endl;
                                    }
                                }

                                std::string kernelKey = run.toHash();
                                kernelData.nrOfConfigs++;
                                kernelData.runs[kernelKey] = std::move(run);
                            }
#ifdef DEBUG_Hist
                            std::cout << "  - Total runs loaded: " << kernelData.runs.size() << std::endl;
#endif
                        }
                        catch(std::exception const& e)
                        {
                            std::cout << e.what() << std::endl;
                        }
                    }

                    kernelData.highestStamp = highestStamp + 1;
                    std::string dataHash = kernelData.toHash();
                    m_tuningHistory[dataHash] = std::move(kernelData);
#ifdef DEBUG_Hist
                    std::cout << "  - KernelData stored with hash: " << dataHash << std::endl;
#endif
                }
            }
            catch(std::exception const& e)
            {
                std::cout << "[DEBUG] Failed to load or parse config file (" << filename << "): " << e.what()
                          << std::endl;
                m_tuningHistory.clear();
            }

#ifdef DEBUG_Hist
            std::cout << "[DEBUG_Hist] Final history size: " << m_tuningHistory.size() << std::endl;
            for(auto const& [k, v] : m_tuningHistory)
            {
                std::cout << "  - Kernel hash: " << k << ", Runs: " << v.runs.size() << std::endl;
                for(auto const& [k1, v1] : v.runs)
                {
                    std::cout << "    - Tuneables: " << v1.tuneables.size() << std::endl;
                }
            }
#endif
        }

        // #define DEBUG_Hist 1

        void storeConfig(std::string const& filename)
        {
            toml::table config;
            for(auto& [key, kernelData] : m_tuningHistory)
            {
                toml::table kernelTable;
#ifdef DEBUG_Hist
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
#ifdef DEBUG_Hist
                std::cout << "Checking kernelData.runs size: " << kernelData.runs.size() << std::endl;
#endif

                int runIndex = 0;
                for(auto& run : kernelData.runs)
                {
                    StorageKernelRun& storeKernel = run.second;
                    if(storeKernel.state == StorageKernelRun::State::Uninitialized)
                    {
                        continue;
                    }
#ifdef DEBUG_Hist
                    std::cout << "Processing m_run #" << runIndex << std::endl;
#endif

                    toml::table runTable;
                    if(storeKernel.state == StorageKernelRun::State::WarmUp)
                    {
                        runTable.emplace("nrRuns", storeKernel.warm_up_runs);
                    }
                    else
                    {
                        runTable.emplace("nrRuns", storeKernel.nr_runs);
                    }


#ifdef DEBUG_Hist
                    std::cout << "  - nrRuns: " << run.second.nr_runs << std::endl;
#endif

                    toml::array metrics;
                    for(auto const& m : run.second.metricContainer.getAll())
                    {
                        metrics.emplace_back(m);
                    }
                    runTable.emplace("metric", metrics);

                    toml::table tuneablesTable;
                    toml::array tuneableNames;
                    toml::array tuneableValues;
#ifdef DEBUG_Hist
                    std::cout << "  - Checking tuneables size: " << run.second.tuneables.size() << std::endl;
#endif
                    for(const auto& tuneable : run.second.view())
                    {
#ifdef DEBUG_Hist

                        std::cout << "    - Adding tuneable: " << tuneable.get().name << " = " << tuneable.get().value
                                  << std::endl;
#endif
                        tuneableNames.emplace_back(tuneable.get().name);
                        tuneableValues.emplace_back(tuneable.get().value);
                    }
                    runTable.emplace("tuneableNames", tuneableNames);
                    runTable.emplace("tuneableVals", tuneableValues);
                    runTable.emplace("stamp", run.second.stamp);

                    runsArray.emplace_back(runTable);

#ifdef DEBUG_Hist
                    std::cout << "Finished processing m_run #" << runIndex << std::endl;
#endif
                    ++runIndex;
                }

#ifdef DEBUG_Hist
                std::cout << "Total runs processed: " << runIndex << std::endl;
#endif
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
