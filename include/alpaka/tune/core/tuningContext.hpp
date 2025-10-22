
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H

#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"
#include "alpaka/tune/utils/compileTimeTemplates.hpp"
#include "alpaka/tune/utils/transformKernelBundle.hpp"
#include "tuningContextManager.hpp"
#include "tuningSession.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/persistentHistory.hpp>
#include <alpaka/tune/IO/runTimeHistory.hpp>
#include <alpaka/tune/core/peripherals/environmentState.hpp>
#include <alpaka/tune/core/peripherals/queue.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>
#include <alpaka/tune/traits/traits.hpp>
#include <alpaka/tune/tunable/kernelTuningModel.hpp>
#include <alpaka/tune/utils/Random.hpp>
#include <alpaka/tune/utils/tupleHelper.hpp>

#include <utility>

namespace alpaka::tune::core

{
    namespace detail
    {
        // transforms the tunable contents of a FrameSpecTune into a tuple


        template<bool B, class U>
        constexpr auto tuple_if(U&& u)
        {
            if constexpr(B)
                return std::tuple{std::forward<U>(u)};
            else
                return std::tuple<>(); // empty
        }

        template<typename T>
        constexpr auto specToFrameTupleHelper(T const& t)
        {
            return std::tuple_cat(
                tuple_if<T::hasNumFramesTune()>(t.getNumFramesTune()),
                tuple_if<T::hasFrameExtentTune()>(t.getFrameExtentTune()),
                tuple_if<T::hasNumBlocksTune()>(t.getNumBlocksTune()),
                tuple_if<T::hasNumThreadsTune()>(t.getNumThreadsTune()));
        }

    } // namespace detail

    /*
     *@TODO reimplement Shrink Tuning Space with new tuneable interface, ensure m_numValues is recalculated,
use model as parameter void shrinkTuningSpace(...)
    {
        while(initialMaxRuns > alpaka::tune::getMaxConfigs())
        {

            // std::vector<std::pair<std::size_t, std::size_t>> maxRunsVec;
            //
            // for_each_enumerate(
            //     expandedTuneables,
            //     [&](auto& wrapper, std::size_t index) { maxRunsVec.emplace_back(wrapper.list.size(), index);
});
            //
            // std::sort(maxRunsVec.begin(), maxRunsVec.end(), std::greater<>());
            //
            // std::size_t maxIndex = maxRunsVec.front().second;
            //
            // bool shouldContinue = true;
            //
            // visitIndex(
            //     maxIndex,
            //     expandedTuneables,
            //     [&](auto& wrapper)
            //     {
            //         if(wrapper.list.size() < 2)
            //         {
            //             shouldContinue = false;
            //             return;
            //         }
            //
            //         bool removed = removeAllMatchingIndices(
            //             allTuneables,
            //             maxIndex,
            //             expandedTuneables,
            //             [](std::size_t i) { return (i & 1) == 1; } // odd indices
            //         );
            //
            //         // fallback if nothing removed
            //         if(!removed)
            //         {
            //             removeAllMatchingIndices(
            //                 allTuneables,
            //                 maxIndex,
            //                 expandedTuneables,
            //                 [](std::size_t i) { return (i & 1) == 0; } // even indices
            //             );
            //         }
            //     });
            //
            // if(!shouldContinue)
            //     break;
            //
            // initialMaxRuns = 1;
            // for_each(allTuneables, [&](auto& tunable) { initialMaxRuns *= tunable.numSteps(); });
        }
#ifdef debug
        printTuneableDimensions(allTuneables, expandedTuneables);
#endif
    }*/

    // #define DEBUG_Singleton
    template<
        typename T_Config,
        typename T_ConfigDescriptor,
        typename T_FrameSpec,
        typename T_Strategy,
        typename T_MetricInterface,
        typename T_Constraints,
        typename T_KernelTuningModel>
    class TuningContext
    {
    public:
        using T_FrameSpecType = T_FrameSpec;
        using T_MetricInterfaceType = T_MetricInterface;
        T_Strategy env_strategy;
        T_MetricInterface env_metricInterface;
        T_Constraints env_constraints;
        T_KernelTuningModel env_kernelTuning;
        IO::KernelTuningMetadata<T_Config, T_ConfigDescriptor> env_kernelData;
        peripherals::EnvironmentState<T_Config> env_environmentState;
        peripherals::ConfigQueue<config::ConfigRecord<T_Config>> env_config_queue;
        TuningContext(TuningContext const&) = delete;
        TuningContext& operator=(TuningContext const&) = delete;
        TuningContext(TuningContext&&) = delete;
        TuningContext& operator=(TuningContext&&) = delete;
        PersistentHistory& env_history;

        auto& getConfigStorage()
        {
            return env_kernelData.configEntries;
        }

        bool violatesConstraint(auto const& config)
        {
            auto& stored = this->getConfigStorage().getOrCreate(config);
            if(stored.state == config::ConfigState::Invalid)
            {
#ifdef Debug
                std::cout << "[violatesConstraint] Already invalid: " << printConfigRecord(stored) << "\n";
#endif
                return true;
            }
            bool valid = true;
            utils::for_each(
                this->env_constraints,
                [&](auto& constraint)
                {
                    if(!constraint.template operator()<T_KernelTuningModel>(this->env_kernelTuning))
                        valid = false;
                });

            if(!valid)
            {
                stored.clearMeasurements(); // clear metric container
                stored.stamp = -1;
                stored.state = config::ConfigState::Invalid;
                stored.fullFlag = true;
                stored.nr_runs = std::numeric_limits<decltype(stored.nr_runs)>::max();
#ifdef Debug
                std::cout << "[violatesConstraint] Marked invalid: " << printConfigRecord(stored) << "\n";
#endif
                ++this->env_environmentState.numberOfCheckedConfigs;
                return true;
            }

            return false;
        }

        TuningContext(
            IO::KernelTuningMetadata<T_Config, T_ConfigDescriptor>&& env_kernelData_,
            T_FrameSpec,
            T_Strategy strategy_,
            T_MetricInterface metric_interface_,
            T_Constraints constraints_,
            T_KernelTuningModel&& kernel_tuning_model_,
            std::string const& filename)
            : env_kernelData(std::forward<IO::KernelTuningMetadata<T_Config, T_ConfigDescriptor>>(env_kernelData_))
            , env_strategy(std::move(strategy_))
            , env_metricInterface(std::move(metric_interface_))
            , env_constraints(std::move(constraints_))
            , env_kernelTuning(std::forward<T_KernelTuningModel>(kernel_tuning_model_))
            , env_history(PersistentHistory::get(filename))

        {
            //@TODO env_history.loadConfig<T_MetricInterface>(env_kernelData, environmentState);

            // Ensure all environment variable–related getters are called (for has... checks)

            getRunsPerConfig();

            getMaxRuns();
            getMaxConfigs();

            auto initConfigTmp = env_kernelTuning.getInitConfig();

            config::ConfigRecord<T_Config>& configEntry = getConfigStorage().getOrCreate(initConfigTmp);
            env_config_queue.push_back(configEntry);

            // Derive maximum counts based on tuning space and environment limits
            env_environmentState.maxConfigsTotal
                = std::min(getMaxCheckConfigs(), env_kernelTuning.getMaxPossibleRuns());
            env_environmentState.maxValidEvaluations = std::min(env_environmentState.maxConfigsTotal, getMaxRuns());
        }

        // Prevent copy/move
    };
    template<typename T>
    struct Dummy;

    template<typename T_Device, typename T_Exec, typename T_FrameSpecTune, typename T_KernelBundle, typename... T_Args>
    auto createTuningEnvironment(
        T_Device device,
        T_Exec exec,
        T_FrameSpecTune const& spec,
        T_KernelBundle bundle,
        alpaka::tune::TuningSession<T_Args...> const& session)
    {
        // Apply HW-specific constraints and adjust frame spaces
        auto newFrameSpecTune = alpaka::tune::adjust::adjustFrameSpecTune(device, exec, spec);

#ifdef DEBUG_Singleton
        newRun.getNumBlocksTune().idxRange.print();
#endif

        // Extract compile-time tuneables for bundle
        auto CTuneableBundle
            = alpaka::tune::CompileTimeHelpers::getCTunables<typename std::remove_cvref_t<T_KernelBundle>::KernelFn>();
        auto userTuple = alpaka::tune::detail::extractTuneables(bundle);
        // Combine into kernel model
        auto completeTuningModel
            = KernelTuningModel{detail::specToFrameTupleHelper(newFrameSpecTune), userTuple, CTuneableBundle};
        auto vals = completeTuningModel.getNumValues();

        using T_completeTuningModel = decltype(completeTuningModel);
        //---- reconfigure kerneltuningModel --- //

        // @TODO shrinkTuningSpace(completeRun.allTuneables(), completeRun.getNumValues()*);
        // Final types deduced for environment
        auto env_kernelData = alpaka::tune::IO::createKernelDataFromModel(
            completeTuningModel,
            alpaka::onHost::demangledName(device),
            alpaka::onHost::demangledName(exec),
            alpaka::onHost::demangledName<T_KernelBundle>(),
            session.m_sessionSpecifier);
        using tuningEnvironmentType = alpaka::tune::core::TuningContext<
            typename decltype(env_kernelData)::TConfig_type,
            decltype(env_kernelData.descriptor),
            decltype(spec.m_spec),
            decltype(session.m_strategy),
            decltype(session.m_metricInterface),
            decltype(session.m_constraintTuple),
            T_completeTuningModel>;
        using T_Context = TuningContextManager<tuningEnvironmentType>;
        return std::make_unique<T_Context>(
            std::move(env_kernelData),
            newFrameSpecTune.m_spec,
            session.m_strategy,
            session.m_metricInterface,
            session.m_constraintTuple,
            std::move(completeTuningModel),
            session.m_outputFile);
    }

    // Static wrapper version
    inline std::string flattenSessionSpecifier(std::vector<std::string> const& vec)
    {
        std::string result;
        for(auto const& s : vec)
        {
            result += s;
        }
        return result;
    }

    template<typename T_Queue, typename T_Exec, typename T_FrameSpecTune, typename T_KernelBundle, typename... T_Args>
    auto& getTuningEnvironment(
        T_Queue const& queue,
        T_Exec const& exec,
        T_FrameSpecTune const& frameSpecTune,
        T_KernelBundle const& bundle,
        alpaka::tune::TuningSession<T_Args...> const& session)
    {
        using EnvPtr = decltype(createTuningEnvironment(queue.getDevice(), exec, frameSpecTune, bundle, session));

        static std::unordered_map<std::string, EnvPtr> singletonMap;


        std::string const key = flattenSessionSpecifier(session.m_sessionSpecifier);
        auto [it, inserted] = singletonMap.try_emplace(
            key,
            createTuningEnvironment(queue.getDevice(), exec, frameSpecTune, bundle, session));

        return it->second;
    }
} // namespace alpaka::tune::core
#endif // KERNELSINGLETON_H
