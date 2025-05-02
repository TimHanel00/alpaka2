//
// Created by tim on 01.05.25.
//

#ifndef TRAITS_HPP
#define TRAITS_HPP
#include <alpaka/alpaka.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>

struct postProcessing
{
    template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename KernelFn, typename... args>
    struct Op
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<KernelFn, args...> const& kernel)
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
            alpaka::KernelBundle<KernelFn, args...> const& kernel)
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
#endif // TRAITS_HPP
