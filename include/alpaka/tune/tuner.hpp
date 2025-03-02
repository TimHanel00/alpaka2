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
namespace alpaka
{

    #define TUNEALL 230
    #define TUNEGRID 1237
    #define TUNETHREADBLOCK 1328
    #include <iostream>
#include <tuple>
#include <string>
#include <unordered_map>
#include <type_traits>
#include "toml.hpp"
#include <toml++/toml.h>


template<typename T_integer=std::size_t, typename T_floating=double_t,typename T_Strategy=tune::strategy::bestRecorded>
struct Tuner
{
    using TuneableValue = std::variant<T_integer, T_floating>;
    using tuneables=std::variant<alpaka::tune::Tuneable<T_integer>,alpaka::tune::Tuneable<T_floating>>;

    T_Strategy strategy{};
    struct KernelRun
    {
        std::vector<tuneables> tuneables;
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
        const T_KernelBundle &bundle;
        KernelRun & run;
        explicit TimeEvent(T_KernelBundle & bundle)
            : startTime(std::chrono::high_resolution_clock::now())
            , bundle(bundle)
            , run()
        {
            auto key = reinterpret_cast<std::uintptr_t>(&bundle);
            if(!activeKernels.contains(key))
            {
                Tuner<T_integer, T_floating>::getInstance().activeKernels[key]
                    = std::make_shared<KernelRun>(KernelRun{});
            }
            run = activeKernels[key];
        }

        // Destructor: Stops the timer and records the duration
        ~TimeEvent()
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
            run->metric=timeDuration.count();
            Tuner<T_integer,T_floating>::getInstance().activeKernels.erase(reinterpret_cast<std::uintptr_t>(&bundle));
        }
    };
    Tuner<T_integer,T_floating> getInstance()
    {
        static Tuner<T_integer,T_floating> tuner;
        return tuner;
    }
    std::unordered_map<std::uintptr_t, std::shared_ptr<KernelRun>> activeKernels;
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
    KernelData getKernelFromHistory(const T_KernelBundle &bundle,const T_Device &device,const T_Executor &executor,const std::string& targetMetric="time")
    {
        std::string lookUpHash=alpaka::core::demangledName<T_Device>(device)+
            alpaka::core::demangledName<T_Executor>(executor)+
                alpaka::core::demangledName<T_KernelBundle>(bundle)+
                    targetMetric;
        if(tuningHistory.contains(lookUpHash))
        {
            return tuningHistory[lookUpHash];
        }
        return nullptr;
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
        std::uintptr_t key = reinterpret_cast<std::uintptr_t>(&KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;

        }
        activeKernels[key]->threadBlockSize=threadBlocktuning;

    }
    template<typename T_KernelBundle,typename T>
    void setGridSizeTuning(const T_KernelBundle & KernelBundle,const alpaka::tune::GridSizeTune<T> & gridSizeTuning)
    {
        std::uintptr_t key = reinterpret_cast<std::uintptr_t>(&KernelBundle);
        if(!activeKernels.contains(key))
        {
            std::cout<<"warning no Time Event for Kernel "<<alpaka::core::demangledName<T_KernelBundle>(KernelBundle)<<"specified"<<std::endl;

        }
        activeKernels[key]->gridSize=gridSizeTuning;
    }
    void loadConfig(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
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
                            std::string name = tuneableArray[0].value_or<std::string>("");
                            auto value = tuneableArray[1].value_or<T_floating>(T_floating{});
                            if(name=="gridSize")
                            {
                                run->gridSize=alpaka::tune::GridSizeTune<T_integer>(static_cast<T_integer>(value));
                                continue;
                            }
                            if(name=="blockThreadSize")
                            {
                                run->threadBlockSize=alpaka::tune::ThreadBlockSizeTune<T_integer>(static_cast<T_integer>(value));
                                continue;
                            }


                            // Create a Tuneable based on the value type (T_integer or T_floating)
                            if constexpr (std::is_same_v<T_floating, T_floating>) { //this comparison is obviously wrong TODO
                                run->tuneables.push_back(alpaka::tune::Tuneable<T_floating>(value, name));
                            } else {
                                run->tuneables.push_back(alpaka::tune::Tuneable<T_integer>(value, name));
                            }
                        }
                    }

                    kernelData.runs.push_back(run);
                }
            }

            tuningHistory[key] = kernelData;
        }
    }
    void storeConfig(const std::string& filename) {
        toml::table config;

        for (const auto& [key, kernelData] : tuningHistory) {
            toml::table kernelTable;
            kernelTable["device"] = kernelData.device;
            kernelTable["executor"] = kernelData.executor;
            kernelTable["kernel"] = kernelData.kernel;
            kernelTable["targetMetric"] = kernelData.targetMetric;

            toml::array runsArray;
            for (const auto& run : kernelData.runs) {
                toml::table runTable;
                runTable["metric"] = run->metric;

                toml::array tuneableArray;
                for (const auto& tuneable : run->tuneables) {
                    std::visit([&tuneableArray](auto&& value) {
                        // Handle each type of TuneableValue (either T_integer or T_floating)
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, alpaka::tune::Tuneable<T_integer>>) {
                            tuneableArray.push_back({value.getName(), value.value}); // Store name and value
                        } else if constexpr (std::is_same_v<T, alpaka::tune::Tuneable<T_floating>>) {
                            tuneableArray.push_back({value.getName(), value.value}); // Store name and value
                        }
                    }, tuneable);
                }
                if(run->threadBlockSize !=std::nullopt)
                {
                    tuneableArray.push_back({run->threadBlockSize.getName(), run->threadBlockSize.value});
                }
                if(run->gridSize !=std::nullopt)
                {
                    tuneableArray.push_back({run->gridSize.getName(), run->gridSize.value});
                }
                runTable["tuneables"] = tuneableArray;
                runsArray.push_back(runTable);
            }

            kernelTable["runs"] = runsArray;
            config[key] = kernelTable;
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
    constexpr bool is_tuneable_v = is_tuneable<T>::value;

    // Updated extractTuneablesImpl function
    template <typename Tuple, typename T_elem, std::size_t... I>
    void extractTuneablesImpl(Tuple&& tup, std::vector<T_elem>& result, std::index_sequence<I...>) {
        // Expand the tuple and check if each element is a Tuneable
        (([&] {
            if constexpr (is_tuneable_v<std::tuple_element_t<I, Tuple>>) {
                auto& element = std::get<I>(tup);

                // Check for the correct type of Tuneable
                using ElementType = std::tuple_element_t<I, Tuple>;

                if constexpr (std::is_same_v<ElementType, alpaka::tune::Tuneable<T_integer>>) {
                    result.push_back(std::variant<alpaka::tune::Tuneable<T_integer>, alpaka::tune::Tuneable<T_floating>>{element});
                }
                else if constexpr (std::is_same_v<ElementType, alpaka::tune::Tuneable<T_floating>>) {
                    result.push_back(std::variant<alpaka::tune::Tuneable<T_integer>, alpaka::tune::Tuneable<T_floating>>{element});
                }
                else {
                    // Throw a runtime exception for invalid type
                    throw std::runtime_error("Error: Invalid Tuneable type encountered.");
                }
            }
        }()), ...);
    }
    template <typename Tuple>
    auto extractTuneables(Tuple&& tup) {
        std::vector<std::variant<alpaka::tune::Tuneable<T_integer>, alpaka::tune::Tuneable<T_floating>>> result;
        extractTuneablesImpl(std::forward<Tuple>(tup), result,
                             std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
        return result;
    }
    template<typename T_KernelBundle>
    void updateArgsWithValues(T_KernelBundle& bundle, const std::vector<tuneables>& extractedTuneables) {
        std::size_t tuneableIndex = 0; // Track position in extractedTuneables

        for (auto& arg : bundle.m_args) {
            if (tuneableIndex >= extractedTuneables.size()) {
                break; // Prevent out-of-bounds errors
            }

            // Check if the argument is a Tuneable<T_integer> or Tuneable<T_floating>
            if (std::holds_alternative<alpaka::tune::Tuneable<T_integer>>(arg)) {
                arg = std::get<alpaka::tune::Tuneable<T_integer>>(extractedTuneables[tuneableIndex++]).value;
            }
            else if (std::holds_alternative<alpaka::tune::Tuneable<T_floating>>(arg)) {
                arg = std::get<alpaka::tune::Tuneable<T_floating>>(extractedTuneables[tuneableIndex++]).value;
            }
            // If it's neither, leave it unchanged
        }
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
        std::vector<tuneables> tuneables=extractTuneables(kernelBundle.m_args);
        std::uintptr_t key = reinterpret_cast<std::uintptr_t>(&kernelBundle);
        auto kernel=getKernelFromHistory(kernelBundle,device,executor);
        if(!kernel)
        {
            kernel=Tuner<T_integer,T_floating>::createKernelData(kernelBundle,device,executor);
            tuningHistory[kernel.toHash()]=kernel;
        }

        auto kernelRunPtr=activeKernels[key];
        kernelRunPtr->tuneables=tuneables;
        T_NumBlocks numblocks=dataBlocking.getThreadSpec().m_numBlocks;
        T_NumThreads numthreads=dataBlocking.getThreadSpec().m_numThreads;
        strategy(kernelRunPtr->tuneables,kernelRunPtr->gridSize,kernelRunPtr->threadBlockSize,kernel.runs,device,executor,kernelBundle);//execute strategy
        if constexpr (kernelRunPtr->gridSize!=std::nullopt){
            numblocks=static_cast<T_NumBlocks>(kernelRunPtr->gridSize.value);
        }
        if constexpr (kernelRunPtr->threadBlockSize!=std::nullopt){
            numthreads=static_cast<T_NumBlocks>(kernelRunPtr->threadBlockSize.value);
        }
        updateArgsWithValues(kernelBundle,kernelRunPtr->tuneables);
        tuningHistory[kernel.toHash()].runs.push_back(kernelRunPtr); //TODO implement logic so that same Parameter Kernel is not Run twice
        return alpaka::onHost::ThreadSpec{numblocks, numthreads};

    }
};
class TunerWrapper
{
private:
    using DefaultTuner = Tuner<>;
    inline static DefaultTuner* instance = nullptr;

public:
    template<typename T_integer = std::size_t, typename T_floating = double,typename T_Strategy=tune::strategy::bestRecorded>
    static DefaultTuner& init() {
        if (!instance) {
            instance = &Tuner<T_integer, T_floating,T_Strategy>::getInstance();
        }
        return *instance;
    }
    static DefaultTuner& get() {
        return *instance;
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
                auto threadSpec=alpaka::TunerWrapper::get().tune(device,executor,dataBlocking,kernelBundle);
                return threadSpec;

            }
}
#endif //TUNER_H
#endif




