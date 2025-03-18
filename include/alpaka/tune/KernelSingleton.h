//
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H
#include "alpaka/core/decay.hpp"
#include "storageTypes.hpp"
#include "tupleHandle.hpp"

#include <alpaka/onHost/FrameSpec.hpp>

#include <utility>

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    typename T_ActiveKernelRun,
    typename T_PtrToHistory,
    typename T_SharedParams,
    bool grid,
    bool block>
class KernelSingleton
{
public:
    using FrameSpecType = alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent>;

    // Public members
    T_Device device;
    T_Exec exec;
    FrameSpecType frameSpec;
    T_KernelBundle kernelBundle;
    T_ActiveKernelRun activeRunPtr;
    T_PtrToHistory ptrToHistory;
    T_SharedParams sharedParams;
    std::vector<std::string> sessionSpecifier;

    // Singleton accessor
    static KernelSingleton& get(
        T_Device device_,
        T_Exec exec_,
        FrameSpecType const& frameSpec_,
        T_KernelBundle const& kernelBundle_,
        T_ActiveKernelRun activeRun_,
        T_PtrToHistory ptrToHistory_,
        T_SharedParams const& sharedParams_,
        std::string const& sessionSpecifier_,
        auto& history,
        auto& run)
    {
        static KernelSingleton instance(
            device_,
            exec_,
            frameSpec_,
            kernelBundle_,
            std::move(activeRun_),
            ptrToHistory_,
            sessionSpecifier_,
            history);
        return instance;
    }

    KernelSingleton(KernelSingleton const&) = delete;
    KernelSingleton& operator=(KernelSingleton const&) = delete;
    KernelSingleton(KernelSingleton&&) = delete;
    KernelSingleton& operator=(KernelSingleton&&) = delete;

private:
    // Private constructor
    KernelSingleton(
        T_Device device_,
        T_Exec exec_,
        FrameSpecType const& frameSpec_,
        T_KernelBundle kernelBundle_,
        T_ActiveKernelRun activeRun_,
        T_PtrToHistory ptrToHistory_,
        T_SharedParams sharedParams_,
        auto sessionSpecifier_,
        auto& history,
        auto& run)
        : device(device_)
        , exec(exec_)
        , kernelBundle(kernelBundle_)
        , activeRunPtr(std::move(activeRun_))
        , ptrToHistory(std::move(ptrToHistory_))
        , sharedParams(std::move(sharedParams_))
        , sessionSpecifier(sessionSpecifier_)
    {
        if(!ptrToHistory)
        {
            // once per tuningSession - make sure to reset static variables since they might persist between
            // multiple instances of TuningSession
            std::string deviceName = alpaka::core::demangledName<T_Device>(device);
            std::string execName = alpaka::core::demangledName<T_Exec>(exec);
            std::string kernelName = alpaka::core::demangledName<T_KernelBundle>(kernelBundle);
            ptrToHistory
                = std::make_shared<KernelData>(createKernelData(deviceName, execName, kernelName, sessionSpecifier));
            std::string key = ptrToHistory->toHash();
            history.m_tuningHistory[key] = std::move(*ptrToHistory);
            ptrToHistory = std::shared_ptr<KernelData>(&history.m_tuningHistory[key], [](KernelData*) {});
        }
        alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> dyna_frameSpec = frameSpec_;

        applyCustomThreadSpec(*activeRunPtr, dyna_frameSpec);
        // acts like a guard only valid configs are used for the device
        alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> spec
            = SessAdjustThreadSpec(device, exec, dyna_frameSpec, *activeRunPtr);
        frameSpec = spec;
    }

    // Prevent copy/move
};

template<
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle,
    bool grid,
    bool block>
static auto createKernelSingleton(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    auto& run,
    auto& sessionSpecifier,
    auto& history)
{
    auto activeRun = ActiveKernelRun{run.gridSize, run.threadBlockSize, extractTuneables(bundle)};
    auto activePtr = std::make_unique<ALPAKA_TYPEOF(activeRun)>(activeRun);
    auto sharedParams = makeSharedParameterInterface<grid, block, ALPAKA_TYPEOF(activeRun)>(*activePtr);
    auto ptrToHistory = history.getKernelFromHistory(device, exec, bundle, sessionSpecifier);
    return KernelSingleton<
        T_Device,
        T_Exec,
        T_NumFrames,
        T_FrameExtent,
        T_KernelBundle,
        ALPAKA_TYPEOF(activePtr),
        ALPAKA_TYPEOF(ptrToHistory),
        ALPAKA_TYPEOF(sharedParams),
        grid,
        block>::get(device, exec, spec, bundle, std::move(activePtr), sharedParams, sessionSpecifier);
}
#endif // KERNELSINGLETON_H
