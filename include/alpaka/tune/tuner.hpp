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

#include <alpaka/tune/tupleHandle.hpp>

#include <alpaka/tune/strategy.hpp>
#include "../../../toml11/include/toml.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
namespace alpaka
{


    //storage container used for history and tuningSession
template<typename T_integer,typename T_floating>
struct KernelRun
{
    std::vector<tune::Tuneable<T_integer>> tuneables;
    std::optional<tune::GridSizeTune<T_integer>> gridSize{std::nullopt};
    std::optional<tune::ThreadBlockSizeTune<T_integer>> threadBlockSize{std::nullopt};
    T_floating metric;
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
            auto const endTime = std::chrono::high_resolution_clock::now();
            auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
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
    template<typename T_integer>
struct TuningHistory
{
    using T_tuneables=alpaka::tune::Tuneable<T_integer>;
    using T_floating=std::double_t;
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
    std::unordered_map<std::string, KernelData> m_tuningHistory;

    TuningHistory()
    {}
KernelData createKernelData(const std::string &exec,const std::string &device,const std::string &bundle,const auto &kernelRun,const std::vector<std::string> & sessionSpecs,const std::string& targetMetric="time")
    {
        KernelData data;
        data.kernel=bundle;
        data.device=device;
        data.executor=exec;
        data.targetMetric=targetMetric;
        data.specifiers=sessionSpecs;
        return data;
    };
    std::shared_ptr<KernelData> getKernelFromHistory(const std::string &exec,const std::string &device,
                                                const std::string &bundle,const auto &kernelRun,const std::vector<std::string> & sessionSpecs,
                                                 const std::string& targetMetric="time")
    {
        std::string lookUpHash = device +
                                 exec +
                                 bundle +
                                 targetMetric +
                                 std::accumulate(sessionSpecs.begin(),
                                                 sessionSpecs.end(),
                                                 std::string());
        if(m_tuningHistory.contains(lookUpHash))
        {
            // Return a non-owning shared_ptr by using a no-op deleter.
            return std::shared_ptr<KernelData>(&m_tuningHistory[lookUpHash],
                                               [](KernelData*) { /* no deletion performed */ });
        }
        return nullptr;
    }

    using ThisTuner = TuningHistory<T_integer>;
    auto createTuningSession()
        {
            // Define an alias for this Tuner type.
            // Construct and return a TuningSession.
            return std::move(TuningSession<T_integer, ThisTuner,alpaka::tune::strategy::randomSearch>(
                *this,       // the tuner reference
                alpaka::tune::strategy::randomSearch{}));
        }
    template<typename T_Strategy>
    auto createTuningSession(T_Strategy strategy)
    {
        // Define an alias for this Tuner type.
        return std::move(TuningSession<T_integer,ThisTuner,T_Strategy>(
            *this,       // the tuner reference
            strategy));
        // Construct and return a TuningSession.
    }


    std::unordered_map<std::uintptr_t,KernelData> kernelEvents;
    ~TuningHistory()
    {
        storeConfig("./config/reduce.toml");
    }

    auto find_str(const auto & table,const auto & key)
    {
        std::string s;
        if (table.contains(key)) {
             s= table.at(key).as_string();
        }
        return s;
    }
    std::string filename;
    template<bool loadMultiple = false>
    void loadConfig(const std::string& filename) {
        try {
            // Parse the TOML file directly from the filename.
            auto config = toml::parse(filename);
            // Retrieve the top-level table.
            const auto& config_table = toml::get<toml::table>(config);

            if (!loadMultiple) {
                std::cout << "[DEBUG] loadMultiple is false, clearing tuning history." << std::endl;
                m_tuningHistory.clear();
            }

            // Iterate over all top-level key/value pairs.
            for (const auto& [key, value] : config_table) {
                // Skip if the value is not a table.
                if (!value.is_table()) continue;
                const auto& kernelTable = value.as_table();

                KernelData kernelData;
                // Retrieve string values safely.
                //kernelData.device       = kernelTable.at("device");
                kernelData.device=find_str(kernelTable, "device");
                kernelData.executor=find_str(kernelTable, "executor");
                kernelData.kernel=find_str(kernelTable, "kernel");
                kernelData.targetMetric=find_str(kernelTable, "targetMetric");

                // Process the "specifiers" array if it exists.
                if (kernelTable.contains("specifiers")) {
                    try {
                        const toml::array& specifiers=kernelTable.at("specifiers").as_array();
                        for (const auto& specifier : specifiers) {
                            if (specifier.is_string()) {
                                kernelData.specifiers.push_back(specifier.as_string());
                            }
                        }
                    } catch (const std::exception&) {
                        // Ignore if "specifiers" is not an array
                    }
                }

                // Process the "runs" array if it exists.
                if (kernelTable.contains("runs")) {
                    try {
                        const auto& runs =kernelTable.at("runs").as_array();
                        for (const auto& runValue : runs)
                        {
                            KernelRun<T_integer,T_floating> run;
                            const auto& runTable = runValue.as_table();
                            run.metric=static_cast<std::size_t>(runTable.at("metric").as_floating());
                            if (runTable.contains("tuneables")) {
                                try {
                                    const auto& tuneables = runTable.at("tuneables").as_table();
                                    for (const auto& [tkey, tval] : tuneables) {
                                            if (tval.is_integer()) {
                                                T_integer value_tuneAble = static_cast<T_integer>(tval.as_integer());
                                                if (tkey == "gridSize") {
                                                    run.gridSize = alpaka::tune::GridSizeTune<T_integer>(value_tuneAble);
                                                } else if (tkey == "blockThreadSize") {
                                                    run.threadBlockSize = alpaka::tune::ThreadBlockSizeTune<T_integer>(value_tuneAble);
                                                } else {
                                                    run.tuneables.push_back(alpaka::tune::Tuneable<T_integer>(value_tuneAble, tkey));
                                                }
                                            }

                                    }
                                } catch (const std::exception& e) {
                                    std::cout<<"exception in passing tuneables "<<e.what()<<std::endl;
                                    // Ignore if "tuneables" is not an array
                                }
                            }

                            std::string kernelKey = run.toHash();
                            kernelData.runs[kernelKey] = std::move(run);
                        }
                    } catch (const std::exception& e) {
                        std::cout<<e.what()<<std::endl;
                        // Ignore if "runs" is not an array
                    }
                }

                std::string dataHash = kernelData.toHash();
                m_tuningHistory[dataHash] = std::move(kernelData);
            }
        } catch (const std::exception& e) {
            std::cout << "[DEBUG] Failed to load or parse config file (" << filename
                      << "): " << e.what() << std::endl;
            m_tuningHistory.clear();
        }
        std::cout<<"history size after load: "<<m_tuningHistory.size()<<std::endl;
    }

    void storeConfig(const std::string& filename) {
        toml::table config;
        std::cout<<" size history: "<<m_tuningHistory.size()<<std::endl;
        for (const auto& [key, kernelData] : m_tuningHistory) {
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
                toml::table tuneablesTable;
#ifdef DEBUG
                std::cout << "  - Checking tuneables size: " << run.second.tuneables.size() << std::endl;
#endif
                for (const auto& tuneable : run.second.tuneables) {
                    tuneablesTable.emplace(tuneable.getName(), toml::value(tuneable.value));
                }

                if (run.second.threadBlockSize != std::nullopt) {
                    tuneablesTable.emplace("blockThreadSize", toml::value(run.second.threadBlockSize->value));
                }
                else {
#ifdef DEBUG
                    std::cout << "  - ThreadBlockSize is std::nullopt" << std::endl;
#endif
                }

                if (run.second.gridSize != std::nullopt) {
                    tuneablesTable.emplace("gridSize", toml::value(run.second.gridSize->value));
                }
                else
                {
#ifdef DEBUG
                    std::cout<<"GridSize is std::nullopt"<<std::endl;
#endif
                }
                runTable.emplace("tuneables", tuneablesTable);

                runsArray.push_back(runTable);
            }
            kernelTable.emplace("runs", runsArray);
            config.emplace(key, kernelTable);
        }

        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        file << toml::format(toml::value(config));

        file.close();
    }
    // Disallow copy and move operations
    TuningHistory(const TuningHistory&) = delete;
    TuningHistory& operator=(const TuningHistory&) = delete;

};
#define MetricUndefined std::numeric_limits<float>::quiet_NaN()
    template<typename T_Integer=std::size_t,typename T_Strategy=alpaka::tune::strategy::randomSearch>
struct TuningSession
{

        using T_Tuner=TuningHistory<T_Integer>;
        using T_floating = double_t;
        KernelRun<T_Integer,T_floating> kernelRun;
        T_Tuner history;
        T_Strategy strategy;
        T_Integer dynamicRuns_Nr;
        std::string deviceName;
        std::string execName;
        std::string kernelName;
        std::vector<std::string> sessionSpecifier;
    TuningSession(){}
    TuningSession(T_Strategy strategy): kernelRun(KernelRun<T_Integer, T_floating>{}),
        strategy(strategy)
    {
        kernelRun.metric=MetricUndefined;
    }
        /*
        TuningSession(const TuningSession&) = delete;
        TuningSession& operator=(const TuningSession&) = delete;
        TuningSession(TuningSession&&) noexcept = default;
        TuningSession& operator=(TuningSession&&) noexcept = default;*/

        TuningSession& withConfig(const std::string &config)
        {
            history.loadConfig(config);
            return *this;
        }
        template<typename... T_Specifiers>
        TuningSession& withRunSpecifiers(T_Specifiers... specifiers)
        {
            processArgs(sessionSpecifier,specifiers...);

            return *this;
        }
        TuningSession& withGridSizeTune()
        {
            kernelRun.gridSize=std::move(alpaka::tune::GridSizeTune<T_Integer>{});
            return *this;
        }
        TuningSession& withGridSizeTune(tune::GridSizeTune<T_Integer> tune)
        {
            kernelRun.gridSize=std::move(tune);
            return *this;
        }
        TuningSession& withBlockSizeTune()
        {
            kernelRun.threadBlockSize=std::move(tune::ThreadBlockSizeTune<T_Integer>{});
            return *this;
        }
        TuningSession& withBlockSizeTune(tune::ThreadBlockSizeTune<T_Integer> tune)
        {
            kernelRun.threadBlockSize=std::move(tune);
            return *this;
        }
        TuningSession& withDynamicRuns(std::size_t runs)
        {
            dynamicRuns_Nr=static_cast<T_Integer>(runs);
            return *this;
        }
        template<typename T>
            void setDynamicRuns(T nrOfRuns)
        {
            dynamicRuns_Nr=static_cast<T_Integer>(nrOfRuns);
        }
        template<typename T_DeviceHandle,typename T_Exec,typename T_NumBlocks,typename T_NumThreads,typename T_KernelRun>
        onHost::FrameSpec<T_NumBlocks, T_NumThreads> SessAdjustThreadSpec(T_DeviceHandle &device,T_Exec exec,const alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> & frameSpec,T_KernelRun &run)
            {
                auto spec = alpaka::tune::adjustThreadSpec(device,exec,frameSpec,run);
                return alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>{T_NumBlocks(frameSpec.m_numFrames),
                            T_NumThreads(frameSpec.m_frameExtent),T_NumBlocks(spec.m_numBlocks),T_NumThreads(spec.m_numThreads)};
            }
        using KernelData=typename ALPAKA_TYPEOF(history)::KernelData;
        /** Enqueue and Execute a kernel for the tuning session will run @DynamicRuns times
             * @param queue the kernel will be executed after all previous work in this queue is finished
             * @param executor description how native worker threads will be mapped and grouped to compute grid layers (blocks,
             * threads).
             * @param specification thread or frame specification which provides a chunked description of the thread or frame
             * index domain
             * @param kernelBundle the compute kernel and there arguments
         */
        template<typename T_Queue,typename T_Exec,typename T_NumFrames,typename T_FrameExtent,typename T_KernelBundle>
        auto enqueue(T_Queue &queue,T_Exec exec,const onHost::FrameSpec<T_NumFrames,T_FrameExtent> &frameSpec,const T_KernelBundle &kernelBundle)
        {

            using Device=ALPAKA_TYPEOF(*queue->m_device);
            kernelName=typeid(T_KernelBundle).name();
            execName=alpaka::core::demangledName<T_Exec>(exec);
            deviceName=alpaka::core::demangledName<Device>(*queue->m_device);
            kernelRun.tuneables = extractTuneables<T_Integer>(kernelBundle);

            alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> dyna_frameSpec=frameSpec;
            alpaka::tune::applyCustomThreadSpec(kernelRun,dyna_frameSpec);
            //acts like a guard only valid configs are used for the device
            alpaka::onHost::FrameSpec<T_NumFrames,T_FrameExtent> spec=SessAdjustThreadSpec(*queue->m_device,exec,dyna_frameSpec,kernelRun);
            auto actualKernel=history.getKernelFromHistory(deviceName,execName,kernelName,kernelRun,sessionSpecifier);

            bool initilized =true;
            if(actualKernel==nullptr)
            {
                actualKernel = std::make_shared<KernelData>(history.createKernelData(deviceName,execName,kernelName,kernelRun,sessionSpecifier));
                if(dynamicRuns_Nr==0)
                {   //no run of kernel recorded and no dynamic runs will be recorded
                    return TuningResult{recreate<T_Integer>(kernelBundle,kernelRun.tuneables),dyna_frameSpec};
                }
                initilized=false;
            }
            if(dynamicRuns_Nr==0&&!actualKernel->runs.contains(kernelRun.toHash()))
            {

                //current user defined parameters havent been used yet
                return TuningResult{recreate<T_Integer>(kernelBundle,kernelRun.tuneables),dyna_frameSpec};
            }
            if(dynamicRuns_Nr>0)
            {
                dynamicRuns(queue,exec,kernelBundle,kernelRun,actualKernel.get(),spec);
            }
            if(!initilized)
            {
                std::string key=actualKernel->toHash();
                history.m_tuningHistory[key]=std::move(*actualKernel);
                actualKernel=std::shared_ptr<KernelData>(&history.m_tuningHistory[key],
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
        template<typename T_KernelBundle,typename T_kernelRun,typename T_NumBlocks,typename T_NumThreads>
            void dynamicRuns(const auto &queue,auto exec,const T_KernelBundle &kernelBundle,T_kernelRun &run,KernelData *data,alpaka::onHost::FrameSpec<T_NumBlocks,T_NumThreads> &spec)
        {
            T_Integer index=0;
            std::vector<std::shared_ptr<tune::Tuneable<T_Integer>>> sharedParams=makeSharedParameterInterface(run);
            while(index<dynamicRuns_Nr)
            {
                strategy(sharedParams,run,data->runs);

                alpaka::tune::applyCustomThreadSpec(run,spec);
                {
                    auto bundle = recreate<T_Integer>(kernelBundle, run.tuneables);
                    auto event=createTimeEvent();
                    onHost::enqueue(queue, exec, spec, bundle);
                    onHost::wait(queue);
                }
                if(!data->runs.contains(run.toHash())){
                    data->runs[run.toHash()]=run;
                }
                ++index;
            }
        }
    ~TuningSession()
    {
            std::cout<<"TuningSession::~TuningSession()\n";
        if(!std::isnan(kernelRun.metric)&&!kernelName.empty())
        {
            auto actualKernel=history.getKernelFromHistory(deviceName,execName,kernelName,kernelRun,sessionSpecifier);
            if(actualKernel==nullptr)
            {
                actualKernel = std::make_shared<KernelData>(history.createKernelData(deviceName,execName,kernelName,kernelRun,sessionSpecifier));
                std::cout<<" created Kernel: "<<actualKernel->toHash()<<"\n";
                std::string key=actualKernel->toHash();
                history.m_tuningHistory[key]=std::move(*actualKernel);
                actualKernel=std::shared_ptr<KernelData>(&history.m_tuningHistory[key],
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


// KernelData stores Tuneable parameters as objects

    /*
 * is holding
 */

}
#endif //TUNER_H
#endif




