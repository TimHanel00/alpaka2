
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H

#include "alpaka/tune/adjust/adjust.hpp"
#include "alpaka/tune/utils/compileTimeTemplates.hpp"
#include "alpaka/tune/utils/transformKernelBundle.hpp"
#include "tuningContextManager.hpp"

#include <alpaka/tune/IO/persistentHistory.hpp>
#include <alpaka/tune/IO/runTimeHistory.hpp>
#include <alpaka/tune/core/peripherals/environmentState.hpp>
#include <alpaka/tune/core/peripherals/queue.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>
#include <alpaka/tune/tunable/kernelTuningModel.hpp>
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

    template<
        typename T_Config,
        typename T_FrameSpec,
        typename T_Strategy,
        typename T_MetricInterface,
        typename T_Constraints,
        typename T_KernelTuningModel>
    class TuningContext
    {
    public:
        using TConfig = T_Config;
        using T_FrameSpecType = T_FrameSpec;
        using T_MetricInterfaceType = T_MetricInterface;
        T_Strategy env_strategy;
        T_MetricInterface env_metricInterface;
        T_Constraints env_constraints;
        T_KernelTuningModel env_tuningModel;
        IO::ActiveHistory<T_Config> env_activeHistory;
        IO::KernelTuningMetadata env_metaData;
        peripherals::EnvironmentState<T_Config> env_environmentState;
        peripherals::ConfigQueue<config::ConfigRecord<T_Config>> env_config_queue;
        TuningContext(TuningContext const&) = delete;
        TuningContext& operator=(TuningContext const&) = delete;
        TuningContext(TuningContext&&) = delete;
        TuningContext& operator=(TuningContext&&) = delete;
        IO::PersistentHistory& env_persistentHistory;

        auto& getHistory()
        {
            return env_activeHistory;
        }

        bool violatesConstraint(config::ConfigRecord<T_Config>& configRecord)
        {
            if(configRecord.state == config::ConfigState::Invalid)
            {
                return true;
            }
            bool valid = true;
            utils::for_each(
                this->env_constraints,
                [&](auto& constraint)
                {
                    if(!constraint.template operator()<T_KernelTuningModel>(this->env_tuningModel))
                        valid = false;
                });

            if(!valid)
            {
                configRecord.clearMeasurements(); // clear metric container
                configRecord.stamp = -1;
                configRecord.state = config::ConfigState::Invalid;
                configRecord.nr_runs = std::numeric_limits<decltype(configRecord.nr_runs)>::max();
                return true;
            }

            return false;
        }

        TuningContext(
            IO::KernelTuningMetadata&& env_metadata,
            T_FrameSpec,
            T_Strategy strategy_,
            T_MetricInterface metric_interface_,
            T_Constraints constraints_,
            T_KernelTuningModel&& kernel_tuning_model_,
            std::string const& filename)
            : env_metaData(std::forward<IO::KernelTuningMetadata>(env_metadata))
            , env_strategy(std::move(strategy_))
            , env_metricInterface(std::move(metric_interface_))
            , env_constraints(std::move(constraints_))
            , env_tuningModel(std::forward<T_KernelTuningModel>(kernel_tuning_model_))
            , env_persistentHistory(IO::PersistentHistory::get(filename))

        {
            if(!env_persistentHistory.m_filename.empty())
                env_persistentHistory.read<T_MetricInterface>(
                    this->env_tuningModel,
                    this->env_activeHistory,
                    this->env_metaData,
                    this->env_environmentState);

            getRunsPerConfig();

            getMaxRuns();
            getMaxConfigs();
            env_environmentState.maxConfigsTotal
                = std::min(getMaxCheckConfigs(), env_tuningModel.getMaxPossibleRuns());
            env_environmentState.maxValidEvaluations = std::min(env_environmentState.maxConfigsTotal, getMaxRuns());


            auto initConfigTmp = env_tuningModel.getInitConfig();

            config::ConfigRecord<T_Config>& configEntry = getHistory().getOrCreate(initConfigTmp);
            if(configEntry.state == config::ConfigState::Uninitialized)
            {
                ++this->env_environmentState.numberOfCheckedConfigs;
                configEntry.state = config::ConfigState::Empty;
                if(!violatesConstraint(configEntry))
                {
                    ++this->env_environmentState.numValidConfigs;
                    env_config_queue.push_back(configEntry);
                }
            }
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
        using T_MetricType = typename alpaka::tune::TuningSession<T_Args...>::MetricType;
        auto env_kernelData = IO::createTuningMetaData(
            alpaka::onHost::demangledName(device),
            alpaka::onHost::demangledName(exec),
            bundle,
            session.m_sessionSpecifiers,
            onHost::demangledName<T_MetricType>());
        using T_Config = config::Config<uint32_t, T_completeTuningModel::numDims>;
        using tuningEnvironmentType = alpaka::tune::core::TuningContext<
            T_Config,
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


        std::string const key = flattenSessionSpecifier(session.m_sessionSpecifiers);
        auto [it, inserted] = singletonMap.try_emplace(
            key,
            createTuningEnvironment(queue.getDevice(), exec, frameSpecTune, bundle, session));

        return it->second;
    }
} // namespace alpaka::tune::core
#endif // KERNELSINGLETON_H
