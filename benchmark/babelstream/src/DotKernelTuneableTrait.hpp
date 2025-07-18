//
// Created by tim on 14.05.25.
//
#pragma once
#include <cstdint>
#include <alpaka/tune/traits/traits.hpp>
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
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Add<CTuneable,Data>>
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
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Mult<CTuneable,Data>>
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
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Triad<CTuneable,Data>>
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
    template<typename CTuneable,typename Data>
    struct CompileTimeTuneableTrait<SimdForEachKernel_Copy<CTuneable,Data>>
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
    std::optional<alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads>> frameSpec = std::nullopt)
{
    static alpaka::onHost::FrameSpec<T_NumFrames, T_NumThreads> local_frameSpec{T_NumFrames{}, T_NumThreads{}};
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
    auto const blockThreadExtentMain_default=512u;
    auto const dotGridBlockExtent_default=1024u;

    template<typename CTuneable,typename Data>
    struct RestKernel;
         template<typename CTuneable,typename Data,typename Vec_2, typename T_Config, typename T_Queue>
    struct GetDefaultImpl<RestKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
            using Vec_1=alpaka::Vec<std::uint32_t, 1>;
            std::uint32_t elementsPerFrameItem = getNumElemPerThread<Data>(queue);
            auto arraySize = accessArraySize<Vec_2>();
            auto numFrames = Vec_1{divExZero(arraySize.x(), blockThreadExtentMain_default * elementsPerFrameItem)};
            auto frameExtent=Vec_1{blockThreadExtentMain_default};
            auto all = config.allTuneables();
            for_each_enumerate(
                all,
                [&](auto& tune, auto i)
                {
                    auto& tuneVal = tune.value;
                    std::string name = tune.name();
                     if(name == "NumBlocksTune")
                    {
                        tuneVal = numFrames;
                    }
                    else if(name == "ThreadBlockTune")
                    {
                        tuneVal = frameExtent;
                    }
                    else if(name == "CTune_0")
                    {
                        tuneVal = Vec_1{elementsPerFrameItem};
                    }
                });
        }};
     template<typename CTuneable,typename Data,typename Vec_2, typename T_Config, typename T_Queue>
    struct GetDefaultImpl<SimdForEachKernel_Copy<CTuneable,Data>, Vec_2, T_Queue, T_Config>
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
           GetDefaultImpl<RestKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>::apply(queue,config);
        }};
         template<typename CTuneable,typename Data,typename Vec_2, typename T_Config, typename T_Queue>
    struct GetDefaultImpl<SimdForEachKernel_Add<CTuneable,Data>, Vec_2, T_Queue, T_Config>
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
           GetDefaultImpl<RestKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>::apply(queue,config);
        }};
        template<typename CTuneable, typename Data, typename Vec_2, typename T_Config, typename T_Queue>
struct GetDefaultImpl<SimdForEachKernel_Triad<CTuneable,Data>, Vec_2, T_Queue, T_Config>
{
    static void apply(T_Queue const& queue, T_Config& config)
    {
        GetDefaultImpl<RestKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>::apply(queue, config);
    }
};
template<typename CTuneable, typename Data, typename Vec_2, typename T_Config, typename T_Queue>
            struct GetDefaultImpl<SimdForEachKernel_Mult<CTuneable,Data>, Vec_2, T_Queue, T_Config>
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
           GetDefaultImpl<RestKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>::apply(queue,config);
        }};
        template<typename CTuneable,typename Data,typename Vec_2, typename T_Config, typename T_Queue>
    struct GetDefaultImpl<DotKernel<CTuneable,Data>, Vec_2, T_Queue, T_Config>
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
            using Vec_1=alpaka::Vec<std::uint32_t, 1>;
            std::uint32_t elementsPerFrameItem = getNumElemPerThread<Data>(queue);
            auto arraySize = accessArraySize<Vec_1>();
            auto numFrames = Vec_1{std::min(
                dotGridBlockExtent_default,
                alpaka::divExZero(arraySize.x(), (static_cast<uint32_t>(blockThreadExtentMain_default) * elementsPerFrameItem)))};

            auto dataBlockingDot = onHost::FrameSpec{numFrames, static_cast<uint32_t>(blockThreadExtentMain_default)};
            auto frameExtent=Vec_1{blockThreadExtentMain_default};
            auto all = config.allTuneables();
            for_each_enumerate(
                all,
                [&](auto& tune, auto i)
                {
                    auto& tuneVal = tune.value;
                    if(std::string name = tune.name(); name == "FrameExtentTune")
                    {
                        tuneVal = frameExtent;
                    }
                    else if(name == "NumBlocksTune")
                    {
                        tuneVal = numFrames;
                    }
                    else if(name == "ThreadBlockTune")
                    {
                        tuneVal = frameExtent;
                    }
                    else if(name == "CTune_0")
                    {
                        tuneVal = Vec_1{elementsPerFrameItem};
                    }
                });
        };
    };
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
            return static_cast<uint32_t>(extent[0] * sizeof(Data));
        }
    };
}
