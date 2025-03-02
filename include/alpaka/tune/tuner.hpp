//
// Created by tim on 05.02.25.
//
#define ENABLE_AUTOTUNE
#include "alpaka/onHost.hpp"

#include <cmath>
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
namespace alpaka
{

    #define TUNEALL 230
    #define TUNEGRID 1237
    #define TUNETHREADBLOCK 1328




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

    };
    struct KernelData
    {
        std::vector<std::shared_ptr<KernelRun>> runs;
        std::string device;
        std::string executor;
        std::string kernel;
        std::string targetMetric;
        std::string toHash()
        {
            return device+executor+kernel+targetMetric;
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
                std::cout<<"create new active Kernel"<<std::endl;
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
    KernelData static createKernelData(const T_KernelBundle &bundle,const T_Device &device,const T_Executor &executor,const std::string& targetMetric="time")
    {
        KernelData data;
        data.kernel=alpaka::core::demangledName<T_KernelBundle>(bundle);
        data.device=alpaka::core::demangledName<T_Device>(device);
        data.executor=alpaka::core::demangledName<T_Executor>(executor);
        data.targetMetric=targetMetric;
        return data;
    };
    template<typename T_KernelBundle,typename T_Device,typename T_Executor>
    std::optional<KernelData> getKernelFromHistory(const T_KernelBundle &bundle,const T_Device &device,const T_Executor &executor,const std::string& targetMetric="time")
    {
        std::string lookUpHash=alpaka::core::demangledName<T_Device>(device)+
            alpaka::core::demangledName<T_Executor>(executor)+
                alpaka::core::demangledName<T_KernelBundle>(bundle)+
                    targetMetric;
        if(tuningHistory.contains(lookUpHash))
        {
            return tuningHistory[lookUpHash];
        }
        return std::nullopt;
    }


    std::unordered_map<std::uintptr_t,KernelData> kernelEvents;
    static Tuner& get() {
        static Tuner instance; // Local static variable that ensures only one instance

        return instance;
    }
    std::string filename;

    template<typename T_KernelBundle,typename T>
    void setThreadBlockTuning(const T_KernelBundle & KernelBundle,const alpaka::tune::ThreadBlockSizeTune<T> & threadBlocktuning)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;

        }
        activeKernels[key]->threadBlockSize=threadBlocktuning;

    }
    template<typename T_KernelBundle,typename T>
    void setGridSizeTuning(const T_KernelBundle & KernelBundle,const alpaka::tune::GridSizeTune<T> & gridSizeTuning)
    {
        auto key = alpaka::core::demangledName<T_KernelBundle>(KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;

        }
        activeKernels[key]->gridSize=gridSizeTuning;
    }
    void loadConfig(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cout<<"could not find specified config :: continue with empty tuning history"<<std::endl;
        tuningHistory.clear();
        return;
        //throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    toml::table config = toml::parse(file);
    tuningHistory.clear();

    for (const auto& [key, value] : config) {
        if (!value.is_table()) continue;
        const toml::table& kernelTable = *value.as_table();

        KernelData kernelData;
        kernelData.device = kernelTable["device"].value_or("");
        kernelData.executor = kernelTable["executor"].value_or("");
        kernelData.kernel = kernelTable["kernel"].value_or("");
        kernelData.targetMetric = kernelTable["targetMetric"].value_or("");

        if (kernelTable.contains("runs") && kernelTable["runs"].is_array()) {
            for (const auto& runValue : *kernelTable["runs"].as_array()) {
                if (!runValue.is_table()) continue;

                const toml::table& runTable = *runValue.as_table();
                auto run = std::make_shared<KernelRun>();
                run->metric = runTable["metric"].value_or(T_floating{});

                if (runTable.contains("tuneables") && runTable["tuneables"].is_array()) {
                    for (const auto& tuneable : *runTable["tuneables"].as_array()) {
                        if (!tuneable.is_array()) continue;

                        const auto& tuneableArray = *tuneable.as_array();
                        std::string name = tuneableArray[0].value_or("");
                        auto value = tuneableArray[1].value_or(T_integer{});
;
                        if(name=="gridSize")
                        {
                            run->gridSize = alpaka::tune::GridSizeTune<T_integer>(static_cast<T_integer>(value));
                            continue;
                        }
                        if(name=="blockThreadSize")
                        {
                            run->threadBlockSize = alpaka::tune::ThreadBlockSizeTune<T_integer>(static_cast<T_integer>(value));
                            continue;
                        }


                        run->tuneables.push_back(alpaka::tune::Tuneable<T_integer>(static_cast<T_integer>(value), name));

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
            toml::table kernelTable;
            kernelTable.emplace("device",kernelData.device);
            kernelTable.emplace("executor",kernelData.executor);
            kernelTable.emplace("kernel",kernelData.kernel);
            kernelTable.emplace("targetMetric",kernelData.targetMetric);

            toml::array runsArray;
            for (const auto& run : kernelData.runs) {
                toml::table runTable;
                runTable.emplace("metric",run->metric);

                toml::array tuneableArray;
                for (const auto& tuneable : run->tuneables) {
                    toml::table keyValue;
                    keyValue.emplace(tuneable.getName(),toml::value(tuneable.value));
                    tuneableArray.push_back(keyValue);
                }
                if(run->threadBlockSize !=std::nullopt)
                {
                    toml::table keyValue;
                    keyValue.emplace("blockThreadSize",toml::value(run->threadBlockSize->blockThreadSize));
                    tuneableArray.push_back(keyValue);
                }
                if(run->gridSize !=std::nullopt)
                {
                    toml::table keyValue;
                    keyValue.emplace("gridSize",toml::value(run->gridSize->gridSize));
                    tuneableArray.push_back(keyValue);
                }
                runTable.emplace("tuneables",tuneableArray);
                runsArray.push_back(runTable);
            }

            kernelTable.emplace("runs",runsArray);
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
    auto createTimeEvent(const T_KernelBundle& kernelBundle)
    {
        return TimeEvent<T_KernelBundle>(kernelBundle);
    }
    template <typename T>
    struct is_tuneable : std::false_type {};

    template <typename T, typename T_End, typename T_Begin, typename T_Stride>
    struct is_tuneable<alpaka::tune::Tuneable<T, T_End, T_Begin, T_Stride>> : std::true_type {};

    template <typename T>
    static constexpr bool is_tuneable_v = is_tuneable<T>::value;

    // Updated extractTuneablesImpl function
    template <typename Tuple, typename T_elem, std::size_t... I>
    void extractTuneablesImpl(Tuple&& tup, std::vector<T_elem>& result, std::index_sequence<I...>) {
        //TODO
    }

    template <typename Tuple>
    auto extractTuneables(Tuple&& tup) {
        std::vector<alpaka::tune::Tuneable<T_integer>> result;
        extractTuneablesImpl(std::forward<Tuple>(tup), result,
                             std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
        return result;
    }
    template<typename T_KernelBundle>
    void updateArgsWithValues(T_KernelBundle& bundle, const std::vector<T_tuneables>& extractedTuneables) {
        //TODO
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
        std::cout<<"in Tune"<<std::endl;
        std::vector<T_tuneables> tuneables=extractTuneables(kernelBundle.m_args);
        std::cout<<"in Tune 2"<<std::endl;
        auto key = alpaka::core::demangledName<T_KernelBundle>(kernelBundle);
        std::cout<<"kernel Name "<<alpaka::core::demangledName<T_KernelBundle>(kernelBundle)<<std::endl;
        std::cout<<"in Tune 3"<<std::endl;
        auto kernel=getKernelFromHistory(kernelBundle,device,executor);
        if(kernel==std::nullopt)
        {
            std::cout<<"zero"<<std::endl;
        }

        std::cout<<"in Tune 4"<<std::endl;
        KernelData actualKernel;
        std::cout<<"in Tune 5"<<std::endl;
        if(kernel==std::nullopt)
        {
            std::cout<<"in Tune 5.5"<<std::endl;
            actualKernel=createKernelData(kernelBundle,device,executor);
            std::cout<<"in Tune 6.5"<<std::endl;
            tuningHistory[actualKernel.toHash()]=actualKernel;
        }
        else
        {
            actualKernel=*kernel;
        }
        std::cout<<"in Tune 7"<<std::endl;
        auto kernelRunPtr=activeKernels[key];
        for(auto& kernels : activeKernels)
        {
            std::cout<<kernels.first<<std::endl;
            std::cout<<kernels.second<<std::endl;
        }
        std::cout<<key<<std::endl;
        std::cout<<kernelRunPtr<<std::endl;
        std::cout<<"in Tune 8"<<std::endl;
        kernelRunPtr->tuneables=tuneables;
        std::cout<<"in Tune 9"<<std::endl;
        T_NumBlocks numblocks=dataBlocking.getThreadSpec().m_numBlocks;
        T_NumThreads numthreads=dataBlocking.getThreadSpec().m_numThreads;
        std::cout<<"in Tune 10"<<std::endl;
        strategy(kernelRunPtr->gridSize,kernelRunPtr->threadBlockSize,kernelRunPtr->tuneables,actualKernel.runs,device,executor);//execute strategy
        std::cout<<"in Tune 11"<<std::endl;
        if (kernelRunPtr->gridSize!=std::nullopt){
            numblocks=static_cast<T_NumBlocks>(kernelRunPtr->gridSize->value);
        }
        if(kernelRunPtr->threadBlockSize!=std::nullopt){
            numthreads=static_cast<T_NumBlocks>(kernelRunPtr->threadBlockSize->value);
        }
        std::cout<<"in Tune 2"<<std::endl;
        updateArgsWithValues(kernelBundle,kernelRunPtr->tuneables);
        tuningHistory[actualKernel.toHash()].runs.push_back(kernelRunPtr); //TODO implement logic so that same Parameter Kernel is not Run twice
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




