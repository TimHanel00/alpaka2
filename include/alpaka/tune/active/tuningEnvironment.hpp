
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H
#include <alpaka/tune/active/Queue.hpp>
#include "../utils/environmentVars.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/active/activeKernel.hpp>
#include <alpaka/tune/traits/traits.hpp>

#include <any>
#include <utility>

struct EnvironmentState
{
    bool sessionFinished{false};
    uint32_t numberOfCheckedConfigs{0};
    uint32_t numValidConfigs{0};
    uint32_t maxValidEvaluations{0};
    uint32_t maxConfigsTotal{0};
    uint32_t stamp{0};
    StorageKernelRun bestConfig;
    alpaka::tune::ConfigQueue<StorageKernelRun> config_queue;
};

// #define DEBUG_Singleton
template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_ActiveKernelRun,
    typename T_PtrToHistory,
    typename T_SharedParams>
class tuningEnvironment
{
public:
    using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>;
    T_Device device;
    T_Exec exec;
    FrameSpecType frameSpec;
    T_KernelBundle kernelBundle;
    T_Strategy env_strategy;
    T_MetricInterface env_metricInterface;
    T_Constraints env_constraints;
    T_ActiveKernelRun activeRunPtr;
    T_PtrToHistory ptrToHistory;
    T_SharedParams sharedParams;
    EnvironmentState environmentState;
    tuningEnvironment(tuningEnvironment const&) = delete;
    tuningEnvironment& operator=(tuningEnvironment const&) = delete;
    tuningEnvironment(tuningEnvironment&&) = delete;
    tuningEnvironment& operator=(tuningEnvironment&&) = delete;

    tuningEnvironment(
        T_Device device_,
        T_Exec exec_,
        FrameSpecType const& frameSpec_,
        T_KernelBundle kernelBundle_,
        T_Strategy strategy_,
        T_MetricInterface metric_interface_,
        T_Constraints constraints_,
        T_ActiveKernelRun activeRun_,
        T_PtrToHistory ptrToHistory_,
        T_SharedParams uniformParamInterface,
        auto sessionSpecifier_,
        auto& history)
        : device(device_)
        , exec(exec_)
        , frameSpec(frameSpec_)
        , kernelBundle(kernelBundle_)
        , env_strategy(std::move(strategy_))
        , env_metricInterface(std::move(metric_interface_))
        , env_constraints(std::move(constraints_))
        , activeRunPtr(std::move(activeRun_))
        , ptrToHistory(std::move(ptrToHistory_))
        , sharedParams(std::move(uniformParamInterface))
    {
        if(!ptrToHistory)
        {
            // once per tuningSession - make sure to reset static variables since they might persist between
            // multiple instances of TuningSession
            std::string deviceName = alpaka::core::demangledName(device);
            std::string execName = alpaka::core::demangledName(exec);
            std::string kernelName = typeid(kernelBundle).name();
            auto tmp = createKernelData(deviceName, execName, kernelName, sessionSpecifier_);
            history.m_tuningHistory.emplace(tmp.toHash(), std::move(tmp));
            ptrToHistory = history.getKernelFromHistory(device, exec, kernelBundle, sessionSpecifier_);
        }
        getRunsPerConfig_Env();
        KernelData& h = *ptrToHistory;
        activeRunPtr->m_strategyState.configStamp = h.highestStamp;
        // acts like a guard only valid configs are used for the device

        alpaka::tune::clampToSpec(frameSpec, *activeRunPtr);

        alpaka::tune::recalculateMaxRuns(*activeRunPtr);

        applyCustomThreadSpec(*activeRunPtr, frameSpec);
        if(!h.runs.contains(activeRunPtr->toHash()))
        {
            h.runs[activeRunPtr->toHash()] = toStore(*activeRunPtr);
        }
        environmentState.bestConfig = h.runs[activeRunPtr->toHash()];
        environmentState.maxConfigsTotal = activeRunPtr->maxRuns;
        environmentState.maxValidEvaluations = getMaxRuns_Env();
        std::cout << " maxRuns from kernel: " << activeRunPtr->maxRuns << " max runs from env" << getMaxRuns_Env()
                  << std::endl;
    }

    // Prevent copy/move
};

template<typename T_Vec>
auto makeConformToTVec(T_Vec const&, alpaka::tune::NoTune const&)
{
    return alpaka::tune::NoTune{};
}

/*
 * ensures that a a user defined tuning conforms to the framespec types and I know its ugly
 *
 */
template<typename T_Vec, typename T_Tuneable>
auto makeConformToTVec(T_Vec const& vec, T_Tuneable& tuneable)
{
    /*
    using Valuetype = T;
    using dimensionTraversePolicy_type = dimensionTraversePolicy;
    static constexpr std::size_t tag = getId<ID>();
    */

    using T_TuneableVec = typename T_Tuneable::ValueType;
    using T_traversePolicy = typename T_Tuneable::dimensionTraversePolicy_type;
    constexpr auto tuneable_ID = T_Tuneable::tag;
    constexpr std::size_t targetDim = T_Vec::dim();
    constexpr std::size_t sourceDim = ALPAKA_TYPEOF(tuneable.value)::dim();

    if constexpr(sourceDim != targetDim)
    {
        if(!tuneable.userDef)
        {
            T_Vec ones = T_Vec::all(1);
            auto ret
                = alpaka::tune::Tuneable<T_Vec, tuneable_ID, T_traversePolicy>(vec, alpaka::IdxRange{ones, vec, ones});
            ret.userDef = false;
            std::cout << " created vector for conformity " << ret.toHash() << std::endl;
            return ret;
        }
        std::string s = std::string(tuneable.name());
        throw std::runtime_error("The dimension of " + s + " must match the dimension of the threadSpec.");
    }
    else
    {
        if(tuneable.userDef)
        {
            // Fallback case when targetDim == sourceDim
            T_Vec value, begin, end, stride;

            for(std::size_t i = 0; i < targetDim; ++i)
            {
                value[i] = tuneable.value[i];
                begin[i] = tuneable.idxRange.m_begin[i];
                end[i] = tuneable.idxRange.m_end[i];
                stride[i] = tuneable.idxRange.m_stride[i];
            }

            auto ret = alpaka::tune::Tuneable<ALPAKA_TYPEOF(value), tuneable_ID, T_traversePolicy>(
                value,
                alpaka::IdxRange{begin, end, stride});
            return ret;
        }

        T_Vec ones = T_Vec::all(1);
        auto ret = alpaka::tune::Tuneable<ALPAKA_TYPEOF(vec), tuneable_ID, T_traversePolicy>(
            vec,
            alpaka::IdxRange{ones, vec, ones});
        ret.userDef = false;
        return ret;
    }
}

template<typename T_frameSpec, typename... T_Args>
auto makeConformToFrameSpec(T_frameSpec& spec, KernelTuningModel<T_Args...>& kernelRun)
{
    // KernelTuningModel m_run;
    // auto h = makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune());
    return makeActiveKernel(
        kernelRun.userTuneables,
        makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune()),

        makeConformToTVec(spec.m_frameExtent, kernelRun.getFrameExtentTune()),
        makeConformToTVec(spec.m_threadSpec.m_numBlocks, kernelRun.getNumBlocksTune()),
        makeConformToTVec(spec.m_threadSpec.m_numThreads, kernelRun.getThreadBlockSizeTune()));
}

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier,
    typename T_History>
auto createTuningEnvironment(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    T_Strategy& strategy,
    T_MetricInterface& metric_interface,
    T_Constraints& constraint,
    T_Run& run,
    T_SessionSpecifier& sessionSpecifier,
    T_History& history)
{
    auto activeRun = makeConformToFrameSpec(spec, run);
    auto retPair = alpaka::tune::applyHwConstraints(device, exec, spec, activeRun);

    auto CTuneableBundle = alpaka::tune::trait::constructRuntimeCtuneablesForActivKernel(bundle);
    auto newFrameSpec = retPair.first;
    auto newRun = retPair.second;
    auto userTuple = extractTuneables(bundle);
    auto completeRun = KernelTuningModel{userTuple, newRun.frameTuneables, CTuneableBundle};
    using T_config = decltype(completeRun.toConfig());
    // static_assert(std::is_same_v<decltype(completeRun), void()>);
#ifdef DEBUG_Singleton
    printRange(newRun.getNumBlocksTune().idxRange);
#endif

    auto activePtr = std::make_unique<ALPAKA_TYPEOF(completeRun)>(completeRun);
    auto sharedParams = makeSharedParameterInterface(*activePtr);
    auto ptrToHistory = history.getKernelFromHistory(device, exec, bundle, sessionSpecifier);
    using tuningEnvironmentType = tuningEnvironment<
        T_Device,
        T_Exec,
        T_FrameExtent,
        T_NumFrames,
        T_KernelBundle,
        T_Strategy,
        T_MetricInterface,
        T_Constraints,
        ALPAKA_TYPEOF(activePtr),
        ALPAKA_TYPEOF(ptrToHistory),
        ALPAKA_TYPEOF(sharedParams)>;

    return std::make_unique<tuningEnvironmentType>(
        device,
        exec,
        newFrameSpec,
        bundle,
        strategy,
        metric_interface,
        constraint,
        std::move(activePtr),
        ptrToHistory,
        std::move(sharedParams),
        sessionSpecifier,
        history);
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

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_Strategy,
    typename T_MetricInterface,
    typename T_Constraints,
    typename T_Run,
    typename T_SessionSpecifier,
    typename T_History>
auto& getTuningEnvironment(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    T_Strategy& strategy,
    T_MetricInterface& metric_interface,
    T_Constraints& constraint,
    T_Run& run,
    T_SessionSpecifier& sessionSpecifier,
    T_History& history)
{
    using tuningEnvironmentType = decltype(createTuningEnvironment(
        device,
        exec,
        spec,
        bundle,
        strategy,
        metric_interface,
        constraint,
        run,
        sessionSpecifier,
        history));

    static std::unordered_map<std::string, tuningEnvironmentType> singletonMap;
    if(auto it = singletonMap.find(flattenSessionSpecifier(sessionSpecifier)); it != singletonMap.end())
    {
        return it->second;
    }

    singletonMap.emplace(
        flattenSessionSpecifier(sessionSpecifier),
        createTuningEnvironment(
            device,
            exec,
            spec,
            bundle,
            strategy,
            metric_interface,
            constraint,
            run,
            sessionSpecifier,
            history));
    return singletonMap.at(flattenSessionSpecifier(sessionSpecifier));
}

#endif // KERNELSINGLETON_H
