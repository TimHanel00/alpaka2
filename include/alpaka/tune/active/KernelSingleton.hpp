
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H
#include "alpaka/core/decay.hpp"
#include "alpaka/tune/adjust/adjust.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/utils/tupleHandle.hpp>

#include <utility>

// #define DEBUG_Singleton

template<typename T_Range>
void printRange(T_Range& range)
{
    std::cout << " begin: " << range.m_begin << std::endl;
    std::cout << " end: " << range.m_end << std::endl;
    std::cout << "stride: " << range.m_stride << std::endl;
}

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_ActiveKernelRun,
    typename T_PtrToHistory,
    typename T_SharedParams>
class KernelSingleton
{
public:
    using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>;
    T_Device device;
    T_Exec exec;
    FrameSpecType frameSpec;
    T_KernelBundle kernelBundle;
    T_ActiveKernelRun activeRunPtr;
    T_PtrToHistory ptrToHistory;
    T_SharedParams sharedParams;

    KernelSingleton(KernelSingleton const&) = delete;
    KernelSingleton& operator=(KernelSingleton const&) = delete;
    KernelSingleton(KernelSingleton&&) = delete;
    KernelSingleton& operator=(KernelSingleton&&) = delete;

    KernelSingleton(
        T_Device device_,
        T_Exec exec_,
        FrameSpecType const& frameSpec_,
        T_KernelBundle kernelBundle_,
        T_ActiveKernelRun activeRun_,
        T_PtrToHistory ptrToHistory_,
        T_SharedParams uniformParamInterface,
        auto sessionSpecifier_,
        auto& history)
        : device(device_)
        , exec(exec_)
        , frameSpec(frameSpec_)
        , kernelBundle(kernelBundle_)
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
        // acts like a guard only valid configs are used for the device

#ifdef DEBUG_Singleton
        std::cout << " gridSize Range: after adjust" << std::endl;
        printRange(activeRunPtr->getNumBlocksTune().idxRange);
        // std::cout << " blockSize Range: after adjust" << std::endl;
        // printRange(activeRunPtr->getThreadBlockSizeTune().idxRange);
#endif
        clampToSpec(frameSpec, *activeRunPtr);
#ifdef DEBUG_Singleton
        std::cout << " gridSize Range: after adjust" << std::endl;
        printRange(activeRunPtr->getNumBlocksTune().idxRange);
        // std::cout << " blockSize Range: after adjust" << std::endl;
        // printRange(activeRunPtr->getThreadBlockSizeTune().idxRange);

#endif

        recalculateMaxRuns(*activeRunPtr);
        applyCustomThreadSpec(*activeRunPtr, frameSpec);
    }

    // Prevent copy/move
};

template<typename T_Vec>
auto makeConformToTVec(T_Vec const&, alpaka::tune::NoTune)
{
    return alpaka::tune::NoTune{};
}

/*
 * ensures that a a user defined tuning conforms to the framespec types and I know its ugly
 *
 */
template<typename T_Vec, template<typename> class T_Tunable, typename T>
T_Tunable<alpaka::Vec<typename T_Vec::type, T_Vec::dim()>> makeConformToTVec(
    T_Vec const& vec,
    T_Tunable<T> const& tuneable)
{
    constexpr std::size_t targetDim = T_Vec::dim();
    constexpr std::size_t sourceDim = ALPAKA_TYPEOF(tuneable.value)::dim();

    if constexpr(targetDim == 1)
    {
        auto ret = T_Tunable(
            T_Vec(tuneable.value),
            alpaka::IdxRange{
                T_Vec(tuneable.idxRange.m_begin[0]),
                T_Vec(tuneable.idxRange.m_end[0]),
                T_Vec(tuneable.idxRange.m_stride[0])});
        return ret;
    }
    else if constexpr(sourceDim != targetDim)
    {
        if(!tuneable.userDef)
        {
            T_Vec ones = T_Vec::all(1);
            auto ret = T_Tunable{vec, alpaka::IdxRange{ones, vec, ones}};
            ret.userDef = false;
            return ret;
        }
        std::string s = tuneable.name;
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

            T_Tunable ret{value, alpaka::IdxRange{begin, end, stride}};
            return ret;
        }

        T_Vec ones = T_Vec::all(1);
        auto ret = T_Tunable{vec, alpaka::IdxRange{ones, vec, ones}};
        ret.userDef = false;
        return ret;
    }
}

template<typename T_frameSpec, typename... T_Args>
auto makeConformToFrameSpec(T_frameSpec& spec, ActiveKernelRun<T_Args...>& kernelRun)
{
    // auto h = makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune());

    return makeActiveKernel(
        kernelRun.userDefTuneables,
        makeConformToTVec(spec.m_numFrames, kernelRun.getNumFramesTune()),

        makeConformToTVec(spec.m_frameExtent, kernelRun.getFrameExtentTune()),
        makeConformToTVec(spec.m_threadSpec.m_numBlocks, kernelRun.getNumBlocksTune()),
        makeConformToTVec(spec.m_threadSpec.m_numThreads, kernelRun.getThreadBlockSizeTune()));
}

template<typename T_Device, typename T_Exec, typename T_NumFrames, typename T_FrameExtent, typename T_KernelBundle>
auto createKernelSingleton(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    auto& run,
    auto& sessionSpecifier,
    auto& history)
{
    auto activeRun = makeConformToFrameSpec(spec, run);
    auto retPair = alpaka::tune::applyHwConstraints(device, exec, spec, activeRun);
    auto newFrameSpec = retPair.first;
    auto newRun = retPair.second;

    /*
     *newRun is a ActiveKernelRun -- check its function signature at compile time: trigger an compilation error where
     *its signature is revealed
     *
     */
// #define DEBUG_Singleton
#ifdef DEBUG_Singleton
    std::cout << " gridSize Range: fromUser" << std::endl;
    printRange(newRun.getNumBlocksTune().idxRange);
    // std::cout << " blockSize Range: fromUser" << std::endl;
    // printRange(newRun.getThreadBlockSizeTune().idxRange);
#endif

    auto activePtr = std::make_unique<ALPAKA_TYPEOF(newRun)>(newRun);
    auto sharedParams = makeSharedParameterInterface(*activePtr);
    auto ptrToHistory = history.getKernelFromHistory(device, exec, bundle, sessionSpecifier);
    using kernelSingletonType = KernelSingleton<
        T_Device,
        T_Exec,
        ALPAKA_TYPEOF(newFrameSpec.m_frameExtent),
        ALPAKA_TYPEOF(newFrameSpec.m_numFrames),
        T_KernelBundle,
        ALPAKA_TYPEOF(activePtr),
        ALPAKA_TYPEOF(ptrToHistory),
        ALPAKA_TYPEOF(sharedParams)>;
    auto singleTon = std::make_unique<kernelSingletonType>(
        device,
        exec,
        newFrameSpec,
        bundle,
        std::move(activePtr),
        ptrToHistory,
        std::move(sharedParams),
        sessionSpecifier,
        history);
    return singleTon;
}
#endif // KERNELSINGLETON_H
