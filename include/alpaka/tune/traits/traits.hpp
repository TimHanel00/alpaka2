//
// Created by tim on 01.05.25.
//

#ifndef TRAITS_HPP
#define TRAITS_HPP

#include <alpaka/alpaka.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>
#include <alpaka/tune/utils/tupleHash.h>

namespace alpaka::tune::trait
{
    struct postProcessing
    {
        template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename KernelFn, typename... args>
        struct Op
        {
            void operator()(
                T_Config& config,
                T_FrameSpec& frame_spec,
                T_Metric& metricInterface,
                KernelBundle<KernelFn, args...> const& kernel)
            {
            }
        };
    };

    template<typename T_KernelBundle, typename T_Config, typename T_FrameSpec, typename T_Metric>
    inline auto callPostProcessing(

        T_Config& config,
        T_FrameSpec& frame_spec,
        T_Metric& metricInterface,
        T_KernelBundle& KernelBundle)
    {
        return postProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(

            config,
            frame_spec,
            metricInterface,
            KernelBundle);
    }

    // default
    struct preProcessing
    {
        template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename KernelFn, typename... args>
        struct Op
        {
            void operator()(
                T_Config& config,
                T_FrameSpec& frame_spec,
                T_Metric& metricInterface,
                KernelBundle<KernelFn, args...> const& kernel)
            {
            }
        };
    };

    /*
    // example specialization for a Kernel HostSideKernel
    template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename... args1, typename... args2>
    struct preProcessing::Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<HostSideKernel<args1...>, args2...>>
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<HostSideKernel<args1...>, args2...> const& kernelBundle)
        {
            std::cout << " special " << std::endl;
        }
    };
    */
    template<typename T_KernelBundle, typename T_Config, typename T_FrameSpec, typename T_Metric>
    auto callPreProcessing(

        T_Config& config,
        T_FrameSpec& frame_spec,
        T_Metric& metricInterface,
        T_KernelBundle const& KernelBundle)
    {
        return preProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(
            config,
            frame_spec,
            metricInterface,
            KernelBundle);
    }

    template<typename T_KernelBundle, typename Vec_2, typename T_Queue, typename T_Config>
    struct GetDefaultImpl
    {
        static void apply(T_Queue const& queue, T_Config& config)
        {
#pragma message("[Warning] No specialization of getDefault for this kernel bundle — check your KernelBundle type!")
            std::cout << " selected wrong SPECIALIZATION " << std::endl;
            // Optional fallback code
            // or
            // error
            // trigger
        }
    };

    template<typename T_KernelBundle, typename Vec_2, typename T_Queue, typename T_Config>
    auto getDefault(T_Queue const& queue, T_Config& config)

    {
        GetDefaultImpl<T_KernelBundle, Vec_2, T_Queue, T_Config>::apply(queue, config);
    }

    template<typename Kernel>
    struct CompileTimeTuneableTrait
    {
        // a Kernel can have several template parameters (some of which may not be tuneable)
        // -- tuned_indices is a CVector that holds a list of positions for tuneables defined in
        // tuneAbleDefinitions()
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        //-> assert the tuned_indicies is of
        static constexpr auto tuneAbleDefinitions()
        {
            return std::tuple{}; // empty tuple, no tunables
        }
    };

    /*
    template<typename... Args>
    struct CompileTimeTuneableTrait<StencilKernel<Args...>>
    {
        static constexpr auto tuned_indices
            = CVec<std::size_t, static_cast<std::size_t>(0), static_cast<std::size_t>(2)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<CVec<int, 0, 0>, CVec<int, 3, 3>, CVec<int, 1, 1>>{};
            constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1, tune2}; // empty tuple, no tunables
        }
    };*/

} // namespace alpaka::tune::trait
#endif // TRAITS_HPP
