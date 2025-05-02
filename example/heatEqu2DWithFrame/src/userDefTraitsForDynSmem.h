//
// Created by tim on 02.05.25.
//

#ifndef USERDEFTRAITSFORDYNSMEM_H
#define USERDEFTRAITSFORDYNSMEM_H
#include "StencilKernel2.hpp"

#include <alpaka/alpaka.hpp>

template<typename T_NumFrames, typename T_NumThreads>
static accessFrameSpec(std::optional<alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads>>& frameSpec)
{
    static alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads> local_frameSpec{T_NumFrames{}, T_NumThreads{}};
    if(frameSpec.has_value())
    {
        local_frameSpec = frameSpec.value();
    }
    return local_frameSpec;
}

template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename... args2>
struct preProcessing::Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<StencilKernel2, args2...>>
{
    void operator()(
        T_Config& config,
        T_FrameSpec& frame_spec,
        T_Metric& metricInterface,
        alpaka::KernelBundle<StencilKernel2, args2...> const& kernelBundle)
    {
        accessFrameSpec(frame_spec);
    }
};
#endif // USERDEFTRAITSFORDYNSMEM_H
