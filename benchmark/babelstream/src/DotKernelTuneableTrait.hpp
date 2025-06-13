//
// Created by tim on 14.05.25.
//
#pragma once
#include <alpaka/alpaka.hpp>

#include <cstdint>
template<typename CTuneable,typename Data>
struct DotKernel;
template<typename CTuneable,typename Data>
struct SimdForEachKernel_Add;
template<typename CTuneable,typename Data>
struct SimdForEachKernel_Mult;
template<typename CTuneable,typename Data>
struct SimdForEachKernel_Copy;
template<typename CTuneable,typename Data>
struct SimdForEachKernel_Triad;
namespace alpaka::tune::trait
{

    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<DotKernel<CTuneable,Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 64>,
                CVec<std::uint32_t, 2>,
                static_cast<std::size_t>(0)>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Add<CTuneable,Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 64>,
                CVec<std::uint32_t, 1>,
                static_cast<std::size_t>(0)>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Mult<CTuneable,Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 64>,
                CVec<std::uint32_t, 1>,
                static_cast<std::size_t>(0)>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Triad<CTuneable,Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 64>,
                CVec<std::uint32_t, 1>,
                static_cast<std::size_t>(0)>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Copy<CTuneable,Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 64>,
                CVec<std::uint32_t, 1>,
                static_cast<std::size_t>(0)>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

} // namespace alpaka::tune::trait
//- -- > make dynamicSharedMem trait for DotKernel
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
    template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename CTuneable,typename Data,typename... args2>
    struct preProcessing::Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<DotKernel<CTuneable,Data>, args2...>>
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<DotKernel<CTuneable,Data>, args2...> const& kernelBundle)
        {
            accessFrameSpec(std::make_optional(frame_spec));
        }
    };
} // namespace alpaka::tune::trait

namespace alpaka::onHost::trait
{
    template<typename CTuneable,typename Data,typename T_NumFrames, typename T_NumThreads>
    struct BlockDynSharedMemBytes<DotKernel<CTuneable,Data>, ThreadSpec<T_NumFrames, T_NumThreads>>
    {
        BlockDynSharedMemBytes(DotKernel<CTuneable,Data> kernel, ThreadSpec<T_NumFrames, T_NumThreads> spec)
        {
        }

        uint32_t operator()(auto const executor, auto const&... args) const
        {
            auto frameSpec = accessFrameSpec<T_NumFrames, T_NumThreads>();
            auto extent = frameSpec.m_frameExtent;
            std::cout<<" returning shared mem" <<static_cast<uint32_t>(extent[0] * sizeof(Data))<<std::endl;
            return static_cast<uint32_t>(extent[0] * sizeof(Data));
        }
    };
}
