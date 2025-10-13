//
// Created by tim on 14.05.25.
//
#pragma once
#include "alpaka/tune/tuneable/Tunable.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/tune/traits/traits.hpp>

#include <cstdint>
template<typename CTuneable, typename Data>
struct DotKernel;
template<typename CTuneable, typename Data>
struct SimdForEachKernel_Add;
template<typename CTuneable, typename Data>
struct SimdForEachKernel_Mult;
template<typename CTuneable, typename Data>
struct SimdForEachKernel_Copy;
template<typename CTuneable, typename Data>
struct SimdForEachKernel_Triad;

namespace alpaka::tune::trait
{

    template<typename CompType, typename Data>
    struct CompileTimeTuneableTrait<DotKernel<CompType, Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                static_cast<std::size_t>(0),
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 4>,
                CVec<std::uint32_t, 8>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 32>,
                CVec<std::uint32_t, 64>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

    template<typename CompType, typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Add<CompType, Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                static_cast<std::size_t>(0),
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 4>,
                CVec<std::uint32_t, 8>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 32>,
                CVec<std::uint32_t, 64>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

    template<typename CompType, typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Mult<CompType, Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                static_cast<std::size_t>(0),
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 4>,
                CVec<std::uint32_t, 8>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 32>,
                CVec<std::uint32_t, 64>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

    template<typename CompType, typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Triad<CompType, Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = CTunable<
                static_cast<std::size_t>(0),
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 4>,
                CVec<std::uint32_t, 8>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 32>,
                CVec<std::uint32_t, 64>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

    template<typename CompType, typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Copy<CompType, Data>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<
                static_cast<std::size_t>(0),
                CVec<std::uint32_t, 1>,
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 4>,
                CVec<std::uint32_t, 8>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 32>,
                CVec<std::uint32_t, 64>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };

} // namespace alpaka::tune::trait

//- -- > make dynamicSharedMem trait for DotKernel
template<typename T_NumFrames, typename T_NumThreads>
static auto accessFrameSpec(
    std::optional<alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads, T_NumThreads>> frameSpec = std::nullopt)
{
    static alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads, T_NumThreads> local_frameSpec{
        T_NumFrames{},
        T_NumThreads{}};
    if(frameSpec.has_value())
    {
        local_frameSpec = frameSpec.value();
    }
    return local_frameSpec;
}

template<typename Vec_2>
static auto accessArraySize(std::optional<Vec_2> arraySize = std::nullopt)
{
    static Vec_2 local_arraySize{};
    if(arraySize.has_value())
    {
        local_arraySize = arraySize.value();
    }
    return local_arraySize;
}

namespace alpaka::tune::trait
{
    auto const blockThreadExtentMain_default = 512u;
    auto const dotGridBlockExtent_default = 1024u;

    template<typename CTuneable, typename Data>
    struct RestKernel;

    template<
        typename T_Config,
        typename T_FrameSpec,
        typename T_Metric,
        typename CompType,
        typename Data,
        typename... args2>
    struct preProcessing::
        Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<DotKernel<CompType, Data>, args2...>>
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<DotKernel<CompType, Data>, args2...> const& kernelBundle)
        {
            accessFrameSpec(std::make_optional(frame_spec));
        }
    };
} // namespace alpaka::tune::trait

namespace alpaka::onHost::trait
{
    template<typename CompType, typename Data, typename T_NumFrames, typename T_NumThreads>
    struct BlockDynSharedMemBytes<DotKernel<CompType, Data>, ThreadSpec<T_NumFrames, T_NumThreads>>
    {
        BlockDynSharedMemBytes(DotKernel<CompType, Data> kernel, ThreadSpec<T_NumFrames, T_NumThreads>)
        {
        }

        uint32_t operator()(auto const executor, auto const&... args) const
        {
            auto frameSpec = accessFrameSpec<T_NumFrames, T_NumThreads>();
            auto extent = frameSpec.m_frameExtent;
            return static_cast<uint32_t>(extent[0] * sizeof(Data));
        }
    };
} // namespace alpaka::onHost::trait
