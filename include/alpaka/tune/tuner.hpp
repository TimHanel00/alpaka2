//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE
#include "alpaka/onHost.hpp"
#include "tunerCpu.hpp"

#include <cmath>
#include <numeric>
#include <variant>
#ifdef ENABLE_AUTOTUNE

#include "tupleHandle.hpp"

#include <alpaka/tune/strategy.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

#include "../../../tomlplusplus/include/toml++/toml.h"
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

    //storage container used for history and tuningSession
    template<typename T_integer,typename T_floating>
    struct KernelRun
{
    std::vector<alpaka::tune::Tuneable<T_integer>> tuneables;
    std::optional<alpaka::tune::GridSizeTune<T_integer>> gridSize{std::nullopt};
    std::optional<alpaka::tune::ThreadBlockSizeTune<T_integer>> threadBlockSize{std::nullopt};
    T_floating metric;
    std::vector<std::string> runSpecifiers;
    std::string toHash()
    {
        std::string m;
        for(auto const &tuneable : tuneables)
        {
            m+=std::to_string(tuneable.value);
        }
        if(threadBlockSize!=std::nullopt)
        {
            m+=std::to_string(threadBlockSize->value);
        }
        if(gridSize!=std::nullopt)
        {
            m+=std::to_string(gridSize->value);
        }
        return m;
    }
};
    template<typename T_KernelRun>
        struct TimeEvent
    {
        std::chrono::high_resolution_clock::time_point startTime;
        T_KernelRun &kernelRun;
        explicit TimeEvent(T_KernelRun & run)
            : startTime(std::chrono::high_resolution_clock::now())
            , kernelRun(run)
        {}

        // Destructor: Stops the timer and records the duration
        ~TimeEvent()
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
            kernelRun.metric=timeDuration.count();
        }
    };
    template<typename T_KernelBundle,typename T_FrameSpec>
    struct TuningResult
    {
        T_KernelBundle m_kernelBundle;
        T_FrameSpec m_frameSpec;
    };
    namespace tune
    {

        template<typename T_KernelRun,typename T_FrameSpec>
    static T_FrameSpec& applyCustomThreadSpec(const T_KernelRun& kernelRun,T_FrameSpec& spec)
        {
            if(kernelRun.gridSize!=std::nullopt)
            {
                spec.m_threadSpec.m_numBlocks=kernelRun.gridSize->value;
            }
            if(kernelRun.threadBlockSize!=std::nullopt)
            {
                spec.m_threadSpec.m_numThreads=kernelRun.threadBlockSize->value;
            }
            return spec;
        }
    }
    template<typename T_Integer,typename T_floating>
    static std::vector<std::shared_ptr<alpaka::tune::Tuneable<T_Integer>>> makeSharedParameterInterface(KernelRun<T_Integer,T_floating>& run)
    {
        std::vector<std::shared_ptr<alpaka::tune::Tuneable<T_Integer>>> ret;

        // Add each tuneable from the vector.
        for(auto& tune : run.tuneables)
        {
            ret.push_back(
                std::shared_ptr<alpaka::tune::Tuneable<T_Integer>>(
                    &tune,
                    [](alpaka::tune::Tuneable<T_Integer>* /*unused*/) { /* no deletion */ }
                )
            );
        }
        if(run.gridSize!=std::nullopt)
        {
            ret.push_back(
                std::shared_ptr<alpaka::tune::Tuneable<T_Integer>>(
                    static_cast<alpaka::tune::Tuneable<T_Integer>*>(&(*run.gridSize)),
                    [](alpaka::tune::Tuneable<T_Integer>* /*unused*/) { /* no deletion */ }
                )
            );
        }
        if(run.threadBlockSize!=std::nullopt)
        {
            ret.push_back(
                std::shared_ptr<alpaka::tune::Tuneable<T_Integer>>(
                    static_cast<alpaka::tune::Tuneable<T_Integer>*>(&(*run.threadBlockSize)),
                    [](alpaka::tune::Tuneable<T_Integer>* /*unused*/) { /* no deletion */ }
                )
            );
        }
        return ret;

    }
    template<typename T_TuningSession>
static auto createTimeEvent(T_TuningSession &session)
    {
        return TimeEvent(session.kernelRun);
    }
    template<typename T_Integer,typename T_floating>
static TimeEvent<KernelRun<T_Integer,T_floating>> createTimeEvent(KernelRun<T_Integer,T_floating> &run)
    {
        return TimeEvent(run);
    }
#define MetricUndefined std::numeric_limits<float>::quiet_NaN()
    template<typename T_Integer,typename T_floating,typename T_NumFrames,typename T_FrameExtent,typename T_Tuner,typename T_KernelBundle,typename T_Strategy>
struct TuningSession
{
    KernelRun<T_Integer,T_floating> kernelRun;
        T_Tuner &tuner;
        const alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> &frameSpec;
        const T_KernelBundle &kernelBundle;
        T_Strategy strategy;
    TuningSession(T_Tuner & tuner,const T_KernelBundle & bundle,const alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> & frameSpec,T_Strategy strategy): kernelRun(KernelRun<T_Integer, T_floating>{})
        , tuner(tuner),frameSpec(frameSpec),kernelBundle(bundle),strategy(strategy)
    {
            kernelRun.metric=MetricUndefined;
    }
        TuningSession(const TuningSession&) = delete;
        TuningSession& operator=(const TuningSession&) = delete;
        TuningSession(TuningSession&&) noexcept = default;
        TuningSession& operator=(TuningSession&&) noexcept = default;

        template<typename... T_Specifiers>
        TuningSession& withRunSpecifiers(T_Specifiers... specifiers)
        {
            processArgs(kernelRun.runSpecifiers,specifiers...);
            kernelRun.tuneables = extractTuneables<T_Integer>(kernelBundle);
            return *this;
        }
        TuningSession& withGridSizeTune(tune::GridSizeTune<T_Integer> tune)
        {
            kernelRun.gridSize=std::move(tune);
            return *this;
        }
        TuningSession& withBlockSizeTune(tune::ThreadBlockSizeTune<T_Integer> tune)
        {
            kernelRun.threadBlockSize=std::move(tune);
            return *this;
        }
        using KernelData=typename ALPAKA_TYPEOF(tuner)::KernelData;
        auto invoke()
        {
            alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> frameSpec=this->frameSpec;
            alpaka::tune::applyCustomThreadSpec(kernelRun,frameSpec);
            //acts like a guard only valid configs are used for the device
            alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> spec=tuner.adjustThreadSpec(frameSpec,kernelRun);
            auto actualKernel=tuner.getKernelFromHistory(kernelRun,kernelBundle);

            bool initilized =true;
            if(actualKernel==nullptr)
            {
                actualKernel = std::make_shared<KernelData>(tuner.createKernelData(kernelRun, kernelBundle));
                if(tuner.numberOfRuns==0)
                {   //no run of kernel recorded and no dynamic runs will be recorded
                    return TuningResult{recreate<T_Integer>(kernelBundle,kernelRun.tuneables),frameSpec};
                }
                initilized=false;
            }
            if(tuner.numberOfRuns==0&&!actualKernel->runs.contains(kernelRun.toHash()))
            {

                //current user defined parameters havent been used yet
                return TuningResult{recreate<T_Integer>(kernelBundle,kernelRun.tuneables),frameSpec};
            }
            if(tuner.numberOfRuns>0)
            {
                tuner.dynamicRuns(kernelBundle,kernelRun,actualKernel.get(),spec,strategy);
            }
            if(!initilized)
            {
                std::string key=actualKernel->toHash();
                tuner.tuningHistory[key]=std::move(*actualKernel);
                actualKernel=std::shared_ptr<KernelData>(&tuner.tuningHistory[key],
                                               [](KernelData*) { /* no deletion performed */ });

                //actualKernel=&tuner.tuningHistory.at(key);
            }
            alpaka::tune::strategy::bestRecorded{}(kernelRun,actualKernel->runs);
            alpaka::tune::applyCustomThreadSpec(kernelRun,spec);

            std::cout<<" Selected GridSize: "<<spec.m_threadSpec.m_numBlocks<<"\n";
            std::cout<<" Selected ThreadBlockSize: "<<spec.m_threadSpec.m_numThreads<<"\n";
            std::cout<<" Time Estimation: "<<kernelRun.metric/1e6<<" ms\n";
            return TuningResult{recreate<T_Integer>(kernelBundle,kernelRun.tuneables),spec};
        }

    ~TuningSession()
    {
        if(!std::isnan(kernelRun.metric))
        {
            auto actualKernel=tuner.getKernelFromHistory(kernelRun,kernelBundle);
            if(actualKernel==nullptr)
            {
                actualKernel = std::make_shared<KernelData>(tuner.createKernelData(kernelRun, kernelBundle));
                std::string key=actualKernel->toHash();
                tuner.tuningHistory[key]=std::move(*actualKernel);
                actualKernel=std::shared_ptr<KernelData>(&tuner.tuningHistory[key],
                                               [](KernelData*) { /* no deletion performed */ });
            }
            if(!actualKernel->runs.contains(kernelRun.toHash()))
            {
                actualKernel->runs[kernelRun.toHash()]=kernelRun;
            }
        }
    }
        auto createTimeEvent()
        {

            return TimeEvent(kernelRun);
        }


};

template<typename T_integer, typename T_floating,typename T_Device,typename T_Exec>
struct Tuner
{
    using T_tuneables=alpaka::tune::Tuneable<T_integer>;

    T_Device& device;
    const T_Exec& exec;

    Tuner(T_Device& device, const T_Exec& exec)
        : device(device), exec(exec)
    {
    }

    template<typename T_NumBlocks,typename T_NumThreads,typename T_KernelRun>
    alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> adjustThreadSpec(const alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> & frameSpec,T_KernelRun &run)
    {
        auto spec = alpaka::tune::adjustThreadSpec(device,exec,frameSpec,run);
        return alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>{T_NumBlocks(frameSpec.m_numFrames),
                    T_NumThreads(frameSpec.m_frameExtent),T_NumBlocks(spec.m_numBlocks),T_NumThreads(spec.m_numThreads)};
    }

    struct KernelData
    {
        std::unordered_map<std::string,KernelRun<T_integer,T_floating>> runs;
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
    KernelData createKernelData(const auto &kernelRun,const T_KernelBundle &bundle,const std::string& targetMetric="time")
    {
        KernelData data;
        data.kernel=typeid(T_KernelBundle).name();
        data.device=alpaka::core::demangledName<T_Device>(device);
        data.executor=alpaka::core::demangledName<T_Exec>(exec);
        data.targetMetric=targetMetric;
        data.specifiers=std::move(kernelRun.runSpecifiers);
        return data;
    };
    template<typename T_KernelBundle>
    std::shared_ptr<KernelData> getKernelFromHistory(const auto &kernelRun,
                                                 const T_KernelBundle &bundle,
                                                 const std::string& targetMetric="time")
    {
        std::string lookUpHash = alpaka::core::demangledName<T_Device>(device) +
                                 alpaka::core::demangledName<T_Exec>(exec) +
                                 typeid(T_KernelBundle).name() +
                                 targetMetric +
                                 std::accumulate(kernelRun.runSpecifiers.begin(),
                                                 kernelRun.runSpecifiers.end(),
                                                 std::string());
        if(tuningHistory.contains(lookUpHash))
        {
            // Return a non-owning shared_ptr by using a no-op deleter.
            return std::shared_ptr<KernelData>(&tuningHistory[lookUpHash],
                                               [](KernelData*) { /* no deletion performed */ });
        }
        return nullptr;
    }
    template<typename T_KernelBundle,typename T_kernelRun,typename T_NumBlocks,typename T_NumThreads,typename T_Strategy>
    void dynamicRuns(const T_KernelBundle &kernelBundle,T_kernelRun &run,KernelData *data,alpaka::onHost::FrameSpec<T_NumBlocks,T_NumThreads> &spec,T_Strategy strategy)
    {
        T_integer index=0;
        std::vector<std::shared_ptr<tune::Tuneable<T_integer>>> sharedParams=makeSharedParameterInterface(run);
        alpaka::onHost::Queue queue = device.makeQueue();
        while(index<numberOfRuns)
        {

            strategy(sharedParams,run,data->runs);

            alpaka::tune::applyCustomThreadSpec(run,spec);
            {
                auto bundle = recreate<T_integer>(kernelBundle, run.tuneables);

                auto event=createTimeEvent(run);
                onHost::enqueue(queue, exec, spec, bundle);
                onHost::wait(queue);
            }
            if(!data->runs.contains(run.toHash())){
                data->runs[run.toHash()]=run;
            }
            ++index;
        }
    }
    using ThisTuner = Tuner<T_integer, T_floating, T_Device, T_Exec>;
    template<typename T_KernelBundle, typename T_NumFrames, typename T_FrameExtent>
    auto createTuningSession(const T_KernelBundle& bundle,
                             const alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>& frameSpec)
        {
            // Define an alias for this Tuner type.

            // Construct and return a TuningSession.
            return std::move(TuningSession<T_integer, T_floating, T_NumFrames, T_FrameExtent, Tuner, T_KernelBundle,alpaka::tune::strategy::randomSearch>(
                *this,       // the tuner reference
                bundle,      // the kernel bundle
                frameSpec,
                alpaka::tune::strategy::randomSearch{}));
        }
    template<typename T_KernelBundle, typename T_NumFrames, typename T_FrameExtent,typename T_Strategy>
    auto createTuningSession(const T_KernelBundle& bundle,
                             const alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>& frameSpec,T_Strategy strategy)
    {
        // Define an alias for this Tuner type.
        return std::move(TuningSession<T_integer, T_floating, T_NumFrames, T_FrameExtent, Tuner, T_KernelBundle,T_Strategy>(
            *this,       // the tuner reference
            bundle,      // the kernel bundle
            frameSpec,
            strategy));
        // Construct and return a TuningSession.
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
                    auto run = std::make_shared<KernelRun<T_integer,T_floating>>();
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

                                T_integer value_tuneAble = val.value_or(T_integer{});

                                if (name == "gridSize") {
                                    run->gridSize = alpaka::tune::GridSizeTune<T_integer>(value_tuneAble);
                                }
                                else if (name == "blockThreadSize") {
                                    run->threadBlockSize = alpaka::tune::ThreadBlockSizeTune<T_integer>(value_tuneAble);
                                }
                                else {
                                    // Any other tuneable
                                    run->tuneables.push_back(alpaka::tune::Tuneable<T_integer>(value_tuneAble, name));
                                }
                            }
                        }
                    }

                    std::string kernelKey=run->toHash();
                    kernelData.runs[kernelKey]=run;
                }
            }

            tuningHistory[kernelData.toHash()] = kernelData;
        }
    }

    void storeConfig(const std::string& filename) {
        toml::table config;

        for (const auto& [key, kernelData] : tuningHistory) {
            toml::table kernelTable;
#ifdef DEBUG
            std::cout << "KernelData properties:" << std::endl;
            std::cout << "  - Device: " << kernelData.device << std::endl;
            std::cout << "  - Executor: " << kernelData.executor << std::endl;
            std::cout << "  - Kernel: " << kernelData.kernel << std::endl;
            std::cout << "  - Target Metric: " << kernelData.targetMetric << std::endl;

            kernelTable.emplace("device", kernelData.device);
            kernelTable.emplace("executor", kernelData.executor);
            kernelTable.emplace("kernel", kernelData.kernel);
            kernelTable.emplace("targetMetric", kernelData.targetMetric);
            std::cout << "Checking kernelData.specifiers size: " << kernelData.specifiers.size() << std::endl;
#endif


            toml::array specifiers_array;
            for (const auto& vec : kernelData.specifiers) {
                specifiers_array.push_back(vec);
            }
            kernelTable.emplace("specifiers", specifiers_array);

            toml::array runsArray;
#ifdef DEBUG
            std::cout << "Checking kernelData.runs size: " << kernelData.runs.size() << std::endl;
#endif
            for (const auto& run : kernelData.runs) {

                toml::table runTable;

                runTable.emplace("metric", run.second.metric);

                toml::array tuneableArray;
#ifdef DEBUG
                std::cout << "  - Checking tuneables size: " << run.second.tuneables.size() << std::endl;
#endif
                for (const auto& tuneable : run.second.tuneables) {
                    toml::table keyValue;
                    keyValue.emplace(tuneable.getName(), toml::value(tuneable.value));
                    tuneableArray.push_back(keyValue);
                }

                if (run.second.threadBlockSize != std::nullopt) {
                    toml::table keyValue;
                    keyValue.emplace("blockThreadSize", toml::value(run.second.threadBlockSize->value));
                    tuneableArray.push_back(keyValue);
                }
                else {
#ifdef DEBUG
                    std::cout << "  - ThreadBlockSize is std::nullopt" << std::endl;
#endif
                }

                if (run.second.gridSize != std::nullopt) {
                    toml::table keyValue;
                    keyValue.emplace("gridSize", toml::value(run.second.gridSize->value));
                    tuneableArray.push_back(keyValue);
                }
                else
                {
#ifdef DEBUG
                    std::cout<<"GridSize is std::nullopt"<<std::endl;
#endif
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
    // Disallow copy and move operations
    Tuner(const Tuner&) = delete;
    Tuner& operator=(const Tuner&) = delete;

};
// KernelData stores Tuneable parameters as objects

    /*
 * is holding
 */

}
#endif //TUNER_H
#endif




