
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

template<
    bool grid,
    bool block,
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

    // Public members
    static constexpr auto m_grid = grid;
    static constexpr auto m_block = block;
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
        frameSpec = alpaka::tune::SessAdjustThreadSpec(device, exec, frameSpec, *activeRunPtr);
        clampToSpec(frameSpec, *activeRunPtr);
        recalculateMaxGridBlockRuns(*activeRunPtr);
        applyCustomThreadSpec(*activeRunPtr, frameSpec);
    }

    // Prevent copy/move
};

template<typename T_Vec>
auto makeConformToTVec(T_Vec const&, std::nullopt_t)
{
    return std::nullopt;
}

/*
 * ensures that a
 */
template<
    typename T_Vec,
    template<typename, typename, typename, typename> class T_Tunable,
    typename T,
    typename T_begin,
    typename T_end,
    typename T_stride>
auto makeConformToTVec(T_Vec const&, std::optional<T_Tunable<T, T_begin, T_end, T_stride>> const& tuneable)
{
    auto actualTuneable = tuneable.value();
    if constexpr(std::is_integral_v<ALPAKA_TYPEOF(actualTuneable.value)> && T_Vec::dim() == 1)
    {
        // if its defined with a size_t
        auto ret = T_Tunable(
            T_Vec(actualTuneable.value),
            alpaka::IdxRange{
                T_Vec(actualTuneable.idxRange.m_begin[0]),
                T_Vec(actualTuneable.idxRange.m_end[0]),
                T_Vec(actualTuneable.idxRange.m_stride[0])});
        return std::optional<ALPAKA_TYPEOF(ret)>(ret);
    }
    else
    {
        if constexpr(T_Vec::dim() != ALPAKA_TYPEOF(actualTuneable.value)::dim())
        {
            std::string s = actualTuneable.name;
            throw std::runtime_error("the Dimension of " + s + " has to comply with the dimension of the threadSpec");
        }
        else
        {
            auto init = alpaka::Vec<typename T_Vec::type, T_Vec::dim()>::all(1);
            auto value = init, begin = init, end = init, stride = init;
            for(auto i = 0; i < T_Vec::dim(); ++i)
            {
                value[i] = actualTuneable.value[i];
                begin[i] = actualTuneable.idxRange.m_begin[i];
                end[i] = actualTuneable.idxRange.m_end[i];
                stride[i] = actualTuneable.idxRange.m_stride[i];
            }
            auto ret = T_Tunable{value, alpaka::IdxRange{begin, end, stride}};
            return std::optional<ALPAKA_TYPEOF(ret)>(ret);
        }
    }
}

template<
    bool grid,
    bool block,
    typename T_Device,
    typename T_Exec,
    typename T_NumFrames,
    typename T_FrameExtent,
    typename T_KernelBundle>
auto createKernelSingleton(
    T_Device device,
    T_Exec exec,
    alpaka::onHost::FrameSpec<T_NumFrames, T_FrameExtent> const& spec,
    T_KernelBundle bundle,
    auto& run,
    auto& sessionSpecifier,
    auto& history)
{
    auto activeRun = ActiveKernelRun{
        makeConformToTVec(spec.m_numFrames, run.gridSize),
        makeConformToTVec(spec.m_frameExtent, run.threadBlockSize),
        extractTuneables(bundle)};
    auto activePtr = std::make_unique<ALPAKA_TYPEOF(activeRun)>(activeRun);
    auto sharedParams = makeSharedParameterInterface<grid, block, ALPAKA_TYPEOF(*activePtr)>(*activePtr);
    auto ptrToHistory = history.getKernelFromHistory(device, exec, bundle, sessionSpecifier);
    using kernelSingletonType = KernelSingleton<
        grid,
        block,
        T_Device,
        T_Exec,
        T_NumFrames,
        T_FrameExtent,
        T_KernelBundle,
        ALPAKA_TYPEOF(activePtr),
        ALPAKA_TYPEOF(ptrToHistory),
        ALPAKA_TYPEOF(sharedParams)>;
    auto singleTon = std::make_unique<kernelSingletonType>(
        device,
        exec,
        spec,
        bundle,
        std::move(activePtr),
        ptrToHistory,
        std::move(sharedParams),
        sessionSpecifier,
        history);
    return singleTon;
}
#endif // KERNELSINGLETON_H
