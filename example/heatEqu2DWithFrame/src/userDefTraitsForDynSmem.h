//
// Created by tim on 02.05.25.
//

#ifndef USERDEFTRAITSFORDYNSMEM_H
#define USERDEFTRAITSFORDYNSMEM_H
#include "StencilKernel2.hpp"

#include <alpaka/alpaka.hpp>

template<typename T_NumFrames, typename T_NumThreads>
static auto accessFrameSpec(
    std::optional<alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads>> frameSpec = std::nullopt)
{
    static alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads> local_frameSpec{T_NumFrames{}, T_NumThreads{}};
    if(frameSpec.has_value())
    {
        local_frameSpec = frameSpec.value();
    }
    return local_frameSpec;
}

namespace alpaka::tune::trait
{
    template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename... args2>
    struct preProcessing::Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<StencilKernel2, args2...>>
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<StencilKernel2, args2...> const& kernelBundle)
        {
            accessFrameSpec(std::make_optional(frame_spec));
        }
    };
} // namespace alpaka::tune::trait

namespace alpaka::onHost::trait
{
    template<typename T_NumFrames, typename T_NumThreads>
    struct BlockDynSharedMemBytes<StencilKernel2, ThreadSpec<T_NumFrames, T_NumThreads>>
    {
        BlockDynSharedMemBytes(StencilKernel2 kernel, ThreadSpec<T_NumFrames, T_NumThreads> spec)
        {
        }

        uint32_t operator()(auto const executor, auto const&... args) const
        {
            auto frameSpec = accessFrameSpec<T_NumFrames, T_NumThreads>();
            auto extent = frameSpec.m_frameExtent;
            std::cout << "setting memory to : " << extent.x() << " " << extent.y() << std::endl;
            return static_cast<uint32_t>((extent.y() + 2) * (extent.x() + 2) * sizeof(double));
        }
    };
} // namespace alpaka::onHost::trait
#endif // USERDEFTRAITSFORDYNSMEM_H
