//
// Created by tim on 05.02.25.
//
#ifndef TUNER_H
#define TUNER_H
#define ENABLE_AUTOTUNE
#include "alpaka/onHost.hpp"
#include "tunerCpu.hpp"
#include "tunerGpu.hpp"

#include <cmath>
#include <numeric>
#include <variant>
#ifdef ENABLE_AUTOTUNE
#    include "../../../toml11/include/toml.hpp"
#    include "KernelSingleton.h"
#    include "environmentVars.h"

#    include <alpaka/tune/storageTypes.hpp>
#    include <alpaka/tune/strategy.hpp>
#    include <alpaka/tune/tupleHandle.hpp>

#    include <chrono>
#    include <iostream>
#    include <string>
#    include <unordered_map>

namespace alpaka
{


    template<typename T_KernelRun>
    struct TimeEvent
    {
        std::chrono::high_resolution_clock::time_point startTime;
        T_KernelRun& kernelRun;

        explicit TimeEvent(T_KernelRun& run) : startTime(std::chrono::high_resolution_clock::now()), kernelRun(run)
        {
        }

        // Destructor: Stops the timer and records the duration
        ~TimeEvent()
        {
            auto const endTime = std::chrono::high_resolution_clock::now();
            auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
            kernelRun.metric = convertToT<ALPAKA_TYPEOF(kernelRun.metric)>(timeDuration.count());
        }
    };

    template<typename TuneableType>
    auto makeNonOwningTuneableTuple(TuneableType& t)
    {
        using ValueType = std::remove_reference_t<TuneableType>;
        using elementType = ALPAKA_TYPEOF(t.value);
        if constexpr(alpaka::isVector_v<elementType>)
        {
            return flatten(t);
        }
        else if constexpr(std::is_integral_v<elementType>)
        {
            return std::tuple(
                alpaka::tune::FlatTuneableHandle<elementType>{
                    t.value,
                    t.name,
                    t.userDef,
                    t.idxRange.m_begin[0],
                    t.idxRange.m_end[0],
                    t.idxRange.m_stride[0]});
        }
        throw std::runtime_error("unrecognized tuneableType");
    }

    template<bool grid, bool block, typename T_ActiveKernel>
    auto makeSharedParameterInterface(T_ActiveKernel& run)
    {
        auto tuneableTuple = std::apply(
            [](auto&... elems) { return std::tuple_cat(makeNonOwningTuneableTuple(elems)...); },
            run.tuneables);
        if constexpr(grid && block)
        {
            auto gridTuple = makeNonOwningTuneableTuple(*run.gridSize);
            auto blockTuple = makeNonOwningTuneableTuple(*run.threadBlockSize);
            return std::tuple_cat(tuneableTuple, gridTuple, blockTuple);
        }
        else if constexpr(grid)
        {
            auto gridTuple = makeNonOwningTuneableTuple(*run.gridSize);
            return std::tuple_cat(tuneableTuple, gridTuple);
        }
        else if constexpr(block)
        {
            auto blockTuple = makeNonOwningTuneableTuple(*run.threadBlockSize);
            return std::tuple_cat(tuneableTuple, blockTuple);
        }
        else
        {
            return tuneableTuple;
        }
    }

    template<typename T_TuningSession>
    static auto createTimeEvent(T_TuningSession& session)
    {
        return TimeEvent(session.kernelRun);
    }

    template<typename T_KernelRun>
    static TimeEvent<T_KernelRun> createTimeEvent(T_KernelRun& run)
    {
        return TimeEvent(run);
    }

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
            std::string lookUpHash = alpaka::core::demangledName<T_DeviceHandle>(device)
                                     + alpaka::core::demangledName<T_Exec>(exec)
                                     + alpaka::core::demangledName<T_KernelBundle>(kernelBundle) + targetMetric
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
#    ifdef DEBUG
            std::cout << "history size after load: " << m_tuningHistory.size() << std::endl;
            for(auto const& [k, v] : m_tuningHistory)
            {
                std::cout << " runs after history " << v.runs.size() << std::endl;
                for(auto const& [k1, v1] : v.runs)
                {
                    std::cout << " tuneable Size: " << v1.tuneables.size() << std::endl;
                }
            }
#    endif
        }

        void storeConfig(std::string const& filename)
        {
            toml::table config;
            for(auto const& [key, kernelData] : m_tuningHistory)
            {
                toml::table kernelTable;
#    ifdef DEBUG
                std::cout << "KernelData properties:" << std::endl;
                std::cout << "  - Device: " << kernelData.device << std::endl;
                std::cout << "  - Executor: " << kernelData.executor << std::endl;
                std::cout << "  - Kernel: " << kernelData.kernel << std::endl;
                std::cout << "  - Target Metric: " << kernelData.targetMetric << std::endl;
                std::cout << "Checking kernelData.specifiers size: " << kernelData.specifiers.size() << std::endl;
#    endif
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
#    ifdef DEBUG
                std::cout << "Checking kernelData.runs size: " << kernelData.runs.size() << std::endl;
#    endif
                for(const auto& run : kernelData.runs)
                {
                    toml::table runTable;
                    runTable.emplace("nrRuns", run.second.nr_runs);
                    runTable.emplace("metric", run.second.metric);
                    toml::table tuneablesTable;
#    ifdef DEBUG
                    std::cout << "  - Checking tuneables size: " << run.second.tuneables.size() << std::endl;
#    endif
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
#    ifdef DEBUG
                        std::cout << "  - ThreadBlockSize is std::nullopt" << std::endl;
#    endif
                    }

                    if(run.second.gridSize != std::nullopt)
                    {
                        tuneablesTable.emplace("gridSize", toml::value(run.second.gridSize->value));
                    }
                    else
                    {
#    ifdef DEBUG
                        std::cout << "GridSize is std::nullopt" << std::endl;
#    endif
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

#    define MetricUndefined std::numeric_limits<float>::quiet_NaN()

    template<
        typename T_Strategy = alpaka::tune::strategy::randomSearch,
        typename T_GridSize = alpaka::tune::GridSizeTune<>,
        typename T_BlockSize = alpaka::tune::ThreadBlockSizeTune<>,
        bool grid = false,
        bool block = false>
    struct TuningSession
    {
        using T_floating = double_t;
        using T_Integer = std::size_t;
        ActiveKernelRun<T_GridSize, T_BlockSize, T_floating> run;
        TuningHistory history;
        using KernelData = typename TuningHistory::KernelData;
        T_Strategy strategy;
        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        StorageKernelRun storeKernel;
        std::vector<std::string> sessionSpecifier;
        TuningSession() = default;

        explicit TuningSession(T_Strategy strategy) : strategy(strategy), reRuns(getReRuns())
        {
            run.metric = MetricUndefined;
        }

        /*
        TuningSession(const TuningSession&) = delete;
        TuningSession& operator=(const TuningSession&) = delete;
        TuningSession(TuningSession&&) noexcept = default;
        TuningSession& operator=(TuningSession&&) noexcept = default;*/
        TuningSession& withReRuns(std::size_t const& reRuns)
        {
            if(getReRuns() != 0)
            {
                this->reRuns = reRuns;
            }
            return *this;
        }

        TuningSession& withConfig(std::string const& config)
        {
            this->config = config;
            return *this;
        }

        template<typename... T_Specifiers>
        TuningSession& withRunSpecifiers(T_Specifiers... specifiers)
        {
            processArgs(sessionSpecifier, specifiers...);

            return *this;
        }

        template<bool copyGridSize = false, bool copyBlockSize = false, typename T_newSession>
        void copy(T_newSession& session)
        {
            session.config = config;
            session.dynamicRuns_Nr = dynamicRuns_Nr;
            session.sessionSpecifier = sessionSpecifier;
            session.run.metric = this->run.metric;
            if constexpr(copyGridSize)
            {
                session.run.gridSize = this->run.gridSize;
            }
            else if constexpr(copyBlockSize)
            {
                session.run.threadBlockSize = this->run.threadBlockSize;
            }
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withGridSizeTune(tune::GridSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSession<T_Strategy, tune::GridSizeTune<T, T_Begin, T_End, T_Stride>, T_BlockSize, true, block> ret{
                strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = tune;
            return ret;
        }

        template<typename T, auto dim>
        auto withGridSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSession<T_Strategy, tune::GridSizeTune<VecType, VecType, VecType, VecType>, T_BlockSize, true, block>
                ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune<VecType, VecType, VecType, VecType>{tune});
            return ret;
        }

        auto withGridSizeTune()
        {
            TuningSession<T_Strategy, tune::GridSizeTune<>, T_BlockSize, true, block> ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune{});
            return ret;
        }

        auto withBlockSizeTune()
        {
            TuningSession<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<>, grid, true> ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune{});
            return ret;
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSession<
                T_Strategy,
                T_GridSize,
                tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>,
                grid,
                true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>{tune});
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withBlockSizeTune(tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSession<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride>, grid, true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = tune;
            return ret;
        }

        auto& withDynamicRuns(std::size_t runs)
        {
            dynamicRuns_Nr = static_cast<T_Integer>(runs);
            return *this;
        }

        template<
            typename T_DeviceHandle,
            typename T_Exec,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelRun>
        onHost::FrameSpec<T_NumBlocks, T_NumThreads> SessAdjustThreadSpec(
            T_DeviceHandle device,
            T_Exec exec,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& frameSpec,
            T_KernelRun& run)
        {
            auto spec = alpaka::tune::adjustThreadSpec(device, exec, frameSpec, run);
            return alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>{
                T_NumBlocks(frameSpec.m_numFrames),
                T_NumThreads(frameSpec.m_frameExtent),
                T_NumBlocks(spec.m_numBlocks),
                T_NumThreads(spec.m_numThreads)};
        }

        /** Enqueue and Execute a kernel for the tuning session will run @DynamicRuns times
         * @param device
         * @param queue the kernel will be executed after all previous work in this queue is finished
         * @param exec
         * @param frameSpec
         * @param executor description how native worker threads will be mapped and grouped to compute grid layers
         * (blocks, threads).
         * @param specification thread or frame specification which provides a chunked description of the thread or
         * frame index domain
         * @param kernelBundle the compute kernel and there arguments
         */
        template<
            typename T_Device,
            typename T_Queue,
            typename T_Exec,
            typename T_NumFrames,
            typename T_FrameExtent,
            typename T_KernelBundle>
        auto enqueue(
            T_Device device,
            T_Queue& queue,
            T_Exec exec,
            onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
            T_KernelBundle const& kernelBundle)
        {
            if(!m_initialized)
            {
                history.loadConfig(config);
            }
            static auto kernel = createKernelSingleton<grid, block>(
                device,
                exec,
                frameSpec,
                kernelBundle,
                this->run,
                sessionSpecifier,
                history);
            auto& activeRun = *kernel.activeRunPtr;
            auto& ptrToHistory = kernel.ptrToHistory;


            if(ptrToHistory->runs.size() > activeRun.maxRuns)
            {
                alpaka::tune::strategy::bestRecorded{}(activeRun, ptrToHistory->runs);
                applyCustomThreadSpec(activeRun, kernel.frameSpec);
                auto bundle = recreate(kernelBundle, activeRun.tuneables);
                onHost::enqueue(queue, exec, kernel.frameSpec, bundle);
                onHost::wait(queue);
                return;
            }
            internal_enqueue(queue, exec, kernelBundle, activeRun, ptrToHistory.get(), kernel.frameSpec);
        }

        template<typename T_KernelBundle, typename T_kernelRun, typename T_NumBlocks, typename T_NumThreads>
        void internal_enqueue(
            auto const& queue,
            auto exec,
            T_KernelBundle const& kernelBundle,
            T_kernelRun& run,
            KernelData* data,
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>& spec)
        {
            if(data->runs.contains(run.toHash()))
            {
                StorageKernelRun storeKernel = data->runs[run.toHash()];
                if(!storeKernel.nr_runs < this->reRuns)
                {
                    auto sharedParams = makeSharedParameterInterface<grid, block, T_kernelRun>(run);
                    strategy(sharedParams, run, data->runs);
                }
            }
            applyCustomThreadSpec(run, spec);
            {
                auto bundle = recreate(kernelBundle, run.tuneables);
                auto event = createTimeEventFromActive(run);
                onHost::enqueue(queue, exec, spec, bundle);
                onHost::wait(queue);
            }

            if(!data->runs.contains(run.toHash()))
            {
                std::cout << "Generated through Dynamic Runs: " << run.toHash() << std::endl;
                data->runs[run.toHash()] = toStore(run);
            }
            else
            {
                StorageKernelRun storeKernel = data->runs[run.toHash()];

                if(storeKernel.nr_runs < this->reRuns)
                {
                    storeKernel.metric
                        = ((this->reRuns - storeKernel.nr_runs) * storeKernel.metric + run.metric) + this->reRuns;
                    ++storeKernel.nr_runs;
                }
            }
        }

        ~TuningSession()
        {
            if(m_initialized)
            {
                history.storeConfig(config);
            }
        }

        template<typename T_ActiveKernelRun>
        static auto createTimeEventFromActive(T_ActiveKernelRun& activeRun)
        {
            return TimeEvent(activeRun);
        }

        auto createTimeEvent()
        {
            return TimeEvent(this->storeKernel);
        }
    };

    // KernelData stores Tuneable parameters as objects

    /*
     * is holding
     */

} // namespace alpaka
#endif // TUNER_H
#endif
