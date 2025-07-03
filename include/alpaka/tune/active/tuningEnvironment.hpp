
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H
#include "../utils/environmentVars.hpp"
#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"
#include "alpaka/tune/utils/Random.h"
#include "alpaka/tune/utils/tupleHelper.h"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/active/Queue.hpp>
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

template<typename TuneablesTuple, typename ExpandedTuple>
void printTuneableDimensions(TuneablesTuple const& allTuneables, ExpandedTuple const& expandedTuneables)
{
    for_each_enumerate(
        expandedTuneables,
        [&](auto const& wrapper, std::size_t)
        {
            visitIndex(
                wrapper.id,
                allTuneables,
                [&](auto const& tuneable)
                {
                    std::cout << tuneable.name() << "_" << wrapper.dim << " = [ ";

                    for(std::size_t i = 0; i < wrapper.list.size(); ++i)
                    {
                        std::cout << wrapper.list[i];
                        if(i + 1 < wrapper.list.size())
                            std::cout << ", ";
                    }

                    std::cout << " ]" << std::endl;
                });
        });
}

template<typename... Tuneables>
void makeListsForAllTuneables(std::tuple<Tuneables...>&& allTuneables)
{
    std::apply(
        [](auto&... tuns)
        {
            (void) std::initializer_list<int>{
                (tuns.valueList = tuns.makeList(), 0)... // discard result
            };
        },
        allTuneables);
}

template<typename... Tuneables>
auto expand(std::tuple<Tuneables...>& tuneables)
{
    return [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        return std::tuple_cat(std::get<Is>(tuneables).expand(Is)...);
    }(std::make_index_sequence<sizeof...(Tuneables)>{});
}

template<typename TuneablesTuple, typename WrapperTuple, typename Predicate>
bool removeAllMatchingIndices(
    TuneablesTuple& allTuneables,
    std::size_t wrapperIndex,
    WrapperTuple& wrappers,
    Predicate shouldRemoveIndex)
{
    bool removedAny = false;

    visitIndex(
        wrapperIndex,
        wrappers,
        [&](auto& wrapper)
        {
            visitIndex(
                wrapper.id,
                allTuneables,
                [&](auto& tuneable)
                {
                    for(std::size_t i = wrapper.list.size(); i-- > 0;)
                    {
                        if(shouldRemoveIndex(i))
                        {
                            if(tuneable.removeIfValid(i, wrapper.dim))
                            {
                                removedAny = true;
                            }
                        }
                    }
                });
        });

    return removedAny;
}

template<typename... Tuneables>
void shrinkTuningSpace(std::tuple<Tuneables...>&& allTuneables, std::size_t initialMaxRuns)
{
    std::cout << initialMaxRuns << " runs initial\n";

    auto expandedTuneables = expand(allTuneables);

    while(initialMaxRuns > alpaka::tune::getMaxConfigs())
    {
        std::vector<std::pair<std::size_t, std::size_t>> maxRunsVec;

        for_each_enumerate(
            expandedTuneables,
            [&](auto& wrapper, std::size_t index) { maxRunsVec.emplace_back(wrapper.list.size(), index); });

        std::sort(maxRunsVec.begin(), maxRunsVec.end(), std::greater<>());

        std::size_t maxIndex = maxRunsVec.front().second;

        bool shouldContinue = true;

        visitIndex(
            maxIndex,
            expandedTuneables,
            [&](auto& wrapper)
            {
                if(wrapper.list.size() < 2)
                {
                    shouldContinue = false;
                    return;
                }

                bool removed = removeAllMatchingIndices(
                    allTuneables,
                    maxIndex,
                    expandedTuneables,
                    [](std::size_t i) { return (i & 1) == 1; } // odd indices
                );

                // fallback if nothing removed
                if(!removed)
                {
                    removeAllMatchingIndices(
                        allTuneables,
                        maxIndex,
                        expandedTuneables,
                        [](std::size_t i) { return (i & 1) == 0; } // even indices
                    );
                }
            });

        if(!shouldContinue)
            break;

        initialMaxRuns = 1;
        for_each(allTuneables, [&](auto& tunable) { initialMaxRuns *= tunable.numSteps(); });
    }
    printTuneableDimensions(allTuneables, expandedTuneables);
}

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
    FrameSpecType const defaultFrameSpec;
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
        FrameSpecType const& oldFrameSpec_,
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
        , defaultFrameSpec(oldFrameSpec_)
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
        KernelData& h = *ptrToHistory;
        std::cout<<" blocks: "<<defaultFrameSpec.m_threadSpec.m_numBlocks<< " vs "<<frameSpec.m_threadSpec.m_numBlocks<<std::endl;
        // acts like a guard only valid configs are used for the device
        alpaka::tune::clampToSpec(device, frameSpec, *activeRunPtr);
        makeListsForAllTuneables(activeRunPtr->allTuneables());
        alpaka::tune::recalculateMaxRuns(*activeRunPtr);
        shrinkTuningSpace(activeRunPtr->allTuneables(), activeRunPtr->maxRuns);

        alpaka::tune::recalculateMaxRuns(*activeRunPtr);
        std::cout << activeRunPtr->maxRuns << " runs after shrink" << std::endl;
        // applyCustomThreadSpec(*activeRunPtr, frameSpec);
        if(!h.runs.contains(activeRunPtr->toHash()))
        {
            h.runs[activeRunPtr->toHash()] = toStore(*activeRunPtr);
        }
        environmentState.bestConfig = h.runs[activeRunPtr->toHash()];
        environmentState.maxConfigsTotal = activeRunPtr->maxRuns;
        environmentState.maxValidEvaluations = alpaka::tune::getMaxRuns();
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
    std::cout << tuneable.value.toString() << std::endl;
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
                = alpaka::tune::Tuneable<T_Vec, tuneable_ID, T_traversePolicy>(alpaka::IdxRange{ones, vec, ones}, vec);
            ret.userDef = false;
            ret.hasRange = tuneable.hasRange;
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
            if constexpr(std::is_convertible_v<T_Vec, T_TuneableVec>)
            {
                auto retTuneable = alpaka::tune::Tuneable<T_Vec, tuneable_ID, T_traversePolicy>(
                    tuneable.idxRange,
                    tuneable.value,
                    tuneable.name());

                retTuneable.inputList.resize(tuneable.inputList.size());
                std::transform(
                    tuneable.inputList.begin(),
                    tuneable.inputList.end(),
                    retTuneable.inputList.begin(),
                    [](auto const& x) { return static_cast<T_Vec>(x); });
                std::cout << " has range ENV" << tuneable.hasRange << std::endl;
                std::cout << " has range IN" << tuneable.hasRange << " " << value.toString() << std::endl;
                retTuneable.hasRange = tuneable.hasRange;
                return retTuneable;
            }
            else
            {
                throw std::runtime_error("Type of frameSpec is not convertible to corresponding tuneables.");
            }
        }

        T_Vec ones = T_Vec::all(1);
        auto ret = alpaka::tune::Tuneable<ALPAKA_TYPEOF(vec), tuneable_ID, T_traversePolicy>(
            alpaka::IdxRange{ones, vec, ones},
            vec);
        ret.userDef = false;
        ret.hasRange = tuneable.hasRange;
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
    retPair.second.printFull();
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
        spec,
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
