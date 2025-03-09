//
// Created by tim on 05.02.25.
//
#define ENABLE_AUTOTUNE
#include "alpaka/onHost.hpp"

#include <cmath>
#include <numeric>
#include <variant>
#ifdef ENABLE_AUTOTUNE
#ifndef TUNER_H
#define TUNER_H
#include <alpaka/tune/strategy.hpp>
#include <chrono>
#include "../../../tomlplusplus/include/toml++/toml.h"
#include <iostream>
#include <tuple>
#include <string>
#include <unordered_map>
#include <type_traits>
#include "tupleHandle.h"
namespace alpaka
{

template<typename T_map,typename T_key>
auto &getKeySafe(const T_map & map,T_key & key)
{
    if(map.find(key)==map.end())
    {
        std::cout<<" active Kernel could not be found, make sure you specified a time or performance Event for the Tuner"<<std::endl;
        throw std::runtime_error("key not found");
    }
    return map.at(key);
}

template<typename T_integer=std::size_t, typename T_floating=double_t,typename T_Strategy=tune::strategy::bestRecorded>
struct Tuner
{
    using T_tuneables=alpaka::tune::Tuneable<T_integer>;

    T_Strategy strategy{};
    Tuner() : strategy{} {};
    static Tuner<T_integer, T_floating, T_Strategy>& getInstance() {
        static Tuner<T_integer, T_floating, T_Strategy> instance;

        return instance;
    }
    struct KernelRun
    {
        std::vector<T_tuneables> tuneables;
        std::optional<alpaka::tune::GridSizeTune<T_integer>> gridSize{std::nullopt};
        std::optional<alpaka::tune::ThreadBlockSizeTune<T_integer>> threadBlockSize{std::nullopt};
        T_floating metric;
        std::vector<std::string> specifiers;
        std::string toHash()
        {
            std::string m="";
            for(auto const &tuneable : tuneables)
            {
                m+=std::to_string(tuneable.value);
            }
            if(threadBlockSize!=std::nullopt)
            {
                m+=std::to_string(threadBlockSize->blockThreadSize);
            }
            if(gridSize!=std::nullopt)
            {
                m+=std::to_string(gridSize->gridSize);
            }
            return m;
        }

    };
    struct KernelData
    {
        std::vector<std::shared_ptr<KernelRun>> runs;
        std::string device;
        std::string executor;
        std::string kernel;
        std::string targetMetric;
        std::vector<std::string> specifiers;
        std::string toHash()
        {
            std::string concatenatedSpecifier=std::accumulate(specifiers.begin(), specifiers.end(), std::string());
            return device+executor+kernel+targetMetric+concatenatedSpecifier;
        }

    };
    std::unordered_map<std::string, KernelData> tuningHistory;
    template<typename T_KernelBundle>
    struct TimeEvent
    {
        std::chrono::high_resolution_clock::time_point startTime;
        std::string kernelDemangled;
        std::shared_ptr<KernelRun> run;
        explicit TimeEvent(const T_KernelBundle & KernelBundle)
            : startTime(std::chrono::high_resolution_clock::now())
            , kernelDemangled(alpaka::core::demangledName<T_KernelBundle>(KernelBundle))
        {
            auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);

            if(!Tuner<>::getInstance().activeKernels.contains(key))
            {
                std::cout<<key<<std::endl;
                Tuner<>::getInstance().activeKernels[key]
                    = std::make_shared<KernelRun>(KernelRun{});
            }
            run = Tuner<>::getInstance().activeKernels[key];
        }

        // Destructor: Stops the timer and records the duration
        ~TimeEvent()
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
            run->metric=timeDuration.count();
            Tuner<>::getInstance().activeKernels.erase(kernelDemangled);
        }
    };
    std::unordered_map<std::string, std::shared_ptr<KernelRun>> activeKernels;
    template<typename T_KernelBundle,typename T_Device,typename T_Executor>
    KernelData static createKernelData(const auto &kernelRun,const T_KernelBundle &bundle,const T_Device &device,const T_Executor &executor,const std::string& targetMetric="time")
    {
        KernelData data;
        data.kernel=alpaka::core::demangledName<T_KernelBundle>(bundle);
        data.device=alpaka::core::demangledName<T_Device>(device);
        data.executor=alpaka::core::demangledName<T_Executor>(executor);
        data.targetMetric=targetMetric;
        data.specifiers=std::move(kernelRun->specifiers);
        return data;
    };
    template<typename T_KernelBundle,typename T_Device,typename T_Executor>
    KernelData* getKernelFromHistory(const auto &kernelRun,const T_KernelBundle &bundle,const T_Device &device,const T_Executor &executor,const std::string& targetMetric="time")
    {
        std::string lookUpHash=alpaka::core::demangledName<T_Device>(device)+
            alpaka::core::demangledName<T_Executor>(executor)+
                alpaka::core::demangledName<T_KernelBundle>(bundle)+
                    targetMetric+std::accumulate(kernelRun->specifiers.begin(),kernelRun->specifiers.end(),std::string());
        if(tuningHistory.contains(lookUpHash))
        {
            return &tuningHistory[lookUpHash];
        }
        return nullptr;
    }
    template<typename T>
    void setDynamicRuns(T nrOfRuns)
    {
        numberOfRuns=static_cast<T_integer>(nrOfRuns);
    }

    std::unordered_map<std::uintptr_t,KernelData> kernelEvents;
    static Tuner& get() {
        static Tuner instance; // Local static variable that ensures only one instance
        instance.numberOfRuns=T_integer(0);
        instance.loadConfig("./config/reduce.toml");
        return instance;
    }
    ~Tuner()
    {
        storeConfig("./config/reduce.toml");
    }
    std::string filename;
    T_integer numberOfRuns;
    template<typename T_KernelBundle,typename T>
    void setThreadBlockTuning(T_KernelBundle & KernelBundle,const alpaka::tune::ThreadBlockSizeTune<T> & threadBlocktuning)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;
            return;
        }
        activeKernels[key]->threadBlockSize=std::optional<alpaka::tune::ThreadBlockSizeTune<T>>(threadBlocktuning);
        std::cout<<activeKernels[key]->gridSize->value<<std::endl;
    }
    template<typename T_KernelBundle,typename T>
    void setGridSizeTuning(T_KernelBundle & KernelBundle,const alpaka::tune::GridSizeTune<T> & gridSizeTuning)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;
            return;
        }
        activeKernels[key]->gridSize=std::optional<alpaka::tune::GridSizeTune<T>>(gridSizeTuning);
    }
    template<typename T_KernelBundle,typename ...T_Specifiers>
    void setRunSpecifiers(const T_KernelBundle & KernelBundle, T_Specifiers... specs)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        std::cout<<" run spec: "<<key<<std::endl;
        auto &kernel=getKeySafe(activeKernels,key);
        processArgs(kernel->specifiers,specs...);

    }
    template<typename T_KernelBundle>
    void disableThreadTuning(const T_KernelBundle & KernelBundle)
    {
        std::cout<<" on disable t Tuning "<<std::endl;
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        std::cout<<" key "<<key<<std::endl;
        auto &kernel=getKeySafe(activeKernels,key);
        kernel->threadBlockSize=std::nullopt;
    }
    template<bool loadMultiple=false>
        void loadConfig(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cout<<"could not find specified config :: continue with empty tuning history"<<std::endl;
            tuningHistory.clear();
            return;
            //throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        toml::table config = toml::parse(file);
        if(!loadMultiple)tuningHistory.clear();

        for (const auto& [key, value] : config) {
            if (!value.is_table()) continue;
            const toml::table& kernelTable = *value.as_table();

            KernelData kernelData;
            kernelData.device = kernelTable["device"].value_or("");
            kernelData.executor = kernelTable["executor"].value_or("");
            kernelData.kernel = kernelTable["kernel"].value_or("");
            kernelData.targetMetric = kernelTable["targetMetric"].value_or("");

            if (kernelTable.contains("specifiers") && kernelTable["specifiers"].is_array())
            {
                for (const auto& specifier : *kernelTable["specifiers"].as_array())
                {
                    std::string spec = specifier.value_or("");
                    kernelData.specifiers.emplace_back(spec);
                }
            }
            if (kernelTable.contains("runs") && kernelTable["runs"].is_array()) {
                for (const auto& runValue : *kernelTable["runs"].as_array()) {
                    if (!runValue.is_table()) continue;

                    const toml::table& runTable = *runValue.as_table();
                    auto run = std::make_shared<KernelRun>();
                    run->metric = runTable["metric"].value_or(T_floating{});


                    if (runTable.contains("tuneables") && runTable["tuneables"].is_array()) {
                        for (const auto& tuneableVal : *runTable["tuneables"].as_array()) {
                            // If it's stored as a table, handle that here
                            if (!tuneableVal.is_table()) {
                                // skip or handle differently
                                continue;
                            }
                            const toml::table& tuneableTable = *tuneableVal.as_table();

                            // Expect exactly one key-value pair per table
                            for (auto&& [key, val] : tuneableTable) {
                                std::string name = key.data();  // The tuneable name

                                T_integer value = val.value_or(T_integer{});

                                if (name == "gridSize") {
                                    run->gridSize = alpaka::tune::GridSizeTune<T_integer>(value);
                                    std::cout << " gridSize discovered" << value<<std::endl;
                                }
                                else if (name == "blockThreadSize") {
                                    run->threadBlockSize = alpaka::tune::ThreadBlockSizeTune<T_integer>(value);
                                }
                                else {
                                    // Any other tuneable
                                    run->tuneables.push_back(alpaka::tune::Tuneable<T_integer>(value, name));
                                }
                            }
                        }
                    }


                    kernelData.runs.push_back(run);
                }
            }

            tuningHistory[kernelData.toHash()] = kernelData;
        }
    }
    void storeConfig(const std::string& filename) {
        toml::table config;

        for (const auto& [key, kernelData] : tuningHistory) {
            std::cout << "Processing key: " << key << std::endl;
            toml::table kernelTable;

            std::cout << "KernelData properties:" << std::endl;
            std::cout << "  - Device: " << kernelData.device << std::endl;
            std::cout << "  - Executor: " << kernelData.executor << std::endl;
            std::cout << "  - Kernel: " << kernelData.kernel << std::endl;
            std::cout << "  - Target Metric: " << kernelData.targetMetric << std::endl;

            kernelTable.emplace("device", kernelData.device);
            kernelTable.emplace("executor", kernelData.executor);
            kernelTable.emplace("kernel", kernelData.kernel);
            kernelTable.emplace("targetMetric", kernelData.targetMetric);

            toml::array specifiers_array;
            std::cout << "Checking kernelData.specifiers size: " << kernelData.specifiers.size() << std::endl;
            for (const auto& vec : kernelData.specifiers) {
                specifiers_array.push_back(vec);
            }
            kernelTable.emplace("specifiers", specifiers_array);

            toml::array runsArray;
            std::cout << "Checking kernelData.runs size: " << kernelData.runs.size() << std::endl;

            for (const auto& run : kernelData.runs) {
                if (!run) {
                    std::cerr << "Error: Encountered a nullptr in kernelData.runs!" << std::endl;
                    continue;
                }

                toml::table runTable;

                runTable.emplace("metric", run->metric);

                toml::array tuneableArray;
                std::cout << "  - Checking tuneables size: " << run->tuneables.size() << std::endl;
                for (const auto& tuneable : run->tuneables) {
                    toml::table keyValue;
                    keyValue.emplace(tuneable.getName(), toml::value(tuneable.value));
                    tuneableArray.push_back(keyValue);
                }

                if (run->threadBlockSize != std::nullopt) {
                    toml::table keyValue;
                    keyValue.emplace("blockThreadSize", toml::value(run->threadBlockSize->blockThreadSize));
                    tuneableArray.push_back(keyValue);
                } else {
                    std::cout << "  - ThreadBlockSize is std::nullopt" << std::endl;
                }

                if (run->gridSize != std::nullopt) {
                    toml::table keyValue;
                    keyValue.emplace("gridSize", toml::value(run->gridSize->gridSize));
                    tuneableArray.push_back(keyValue);
                }
                runTable.emplace("tuneables", tuneableArray);

                runsArray.push_back(runTable);
            }

            kernelTable.emplace("runs", runsArray);
            config.emplace(key, kernelTable);
        }

        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        file << config;

        file.close();
    }
    template<typename T_KernelBundle>
    auto createTimeEvent(T_KernelBundle& kernelBundle)
    {
        return TimeEvent<T_KernelBundle>(kernelBundle);
    }
    // Disallow copy and move operations
    Tuner(const Tuner&) = delete;
    Tuner& operator=(const Tuner&) = delete;
    template<typename T_Device,
        typename T_Executor,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelBundle>
        auto tune(
                T_Device const& device,
                T_Executor const& executor,
                alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle & kernelBundle)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(kernelBundle);
        auto &kernelRun=getKeySafe(activeKernels,key);
        KernelData *actualKernel=getKernelFromHistory(kernelRun,kernelBundle,device,executor);
        if(actualKernel==nullptr)
        {
            KernelData create=createKernelData(kernelRun,kernelBundle,device,executor);
            std::string key=create.toHash();
            tuningHistory[key]=std::move(create);
            actualKernel=&tuningHistory.at(key);

        }
        T_NumBlocks numblocks=dataBlocking.getThreadSpec().m_numBlocks;
        T_NumThreads numthreads=dataBlocking.getThreadSpec().m_numThreads;
        if(!activeKernels.contains(key))
        {
            return alpaka::onHost::ThreadSpec{numblocks, numthreads};
        }
        auto kernelRunPtr=activeKernels[key];
        //kernelRunPtr->tuneables
        auto extracted = extractTuneables<T_integer>(kernelBundle);
        T_integer tuneableStopIndex=0;
        if(kernelRunPtr->threadBlockSize)
        {
            extracted.push_back(
            std::shared_ptr<alpaka::tune::Tuneable<T_integer>>(
                static_cast<alpaka::tune::Tuneable<T_integer>*>(&(*kernelRunPtr->threadBlockSize)),
                [](alpaka::tune::Tuneable<T_integer>* /*unused*/) { /* no deletion; non-owning */ }
                )
            );
            tuneableStopIndex+=1;
        }
        if(kernelRunPtr->gridSize)
        {
            extracted.push_back(
            std::shared_ptr<alpaka::tune::Tuneable<T_integer>>(
                static_cast<alpaka::tune::Tuneable<T_integer>*>(&(*kernelRunPtr->gridSize)),
                [](alpaka::tune::Tuneable<T_integer>* /*unused*/) { /* no deletion; non-owning */ }
                )
            );
            tuneableStopIndex+=1;
        }
        strategy(extracted,actualKernel->runs,device,executor);//execute strategy
        std::string currentKernelHash=actualKernel->toHash();
        if (kernelRunPtr->gridSize!=std::nullopt){
            numblocks=T_NumBlocks(kernelRunPtr->gridSize->value);
        }
        if(kernelRunPtr->threadBlockSize!=std::nullopt){
            numthreads=T_NumBlocks(kernelRunPtr->threadBlockSize->value);
        }
        for(auto i=T_integer(0);i<extracted.size()-tuneableStopIndex;i++)
        {
            auto copyOfTuneable=extracted[i]->copy();
            kernelRunPtr->tuneables.emplace_back(copyOfTuneable);
        }
        //ensures same parameter definitions are not used
        for(auto const& run:tuningHistory[currentKernelHash].runs)
        {
            if(run->toHash()==kernelRunPtr->toHash())
                return alpaka::onHost::ThreadSpec{numblocks, numthreads};

        }

        actualKernel->runs.push_back(kernelRunPtr);
        return alpaka::onHost::ThreadSpec{numblocks, numthreads};

    }
};
// KernelData stores Tuneable parameters as objects

    /*
 * is holding
 */



    // Trait to detect Tuneable<T>


    template<typename T_Device,
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelBundle>
        auto tuneWithContext(
                T_Device const& device,
                T_Mapping const& executor,
                alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle & kernelBundle){

                //Tuning steps that are general
                auto threadSpec = Tuner<>::getInstance().tune(device,executor,dataBlocking,kernelBundle);
                return threadSpec;

            }
}
#endif //TUNER_H
#endif




