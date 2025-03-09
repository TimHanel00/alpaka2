//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include "alpaka/api/cpu/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"
#include "tuner.hpp"
namespace alpaka::onHost::internal
{
    template<typename T_Tuner,
        typename T_Platform, // <--- Declare T_Platform properly here!
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelBundle>
    struct TunerAdjustThreadSpec
    {
        auto operator()(
               cpu::Device<T_Platform> const& device,
               T_Mapping const& executor,
               FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
               T_KernelBundle const& kernelBundle,T_Tuner & tuner) const requires alpaka::concepts::CVector<T_NumThreads>
        {
            tuner.disableThreadTuning(kernelBundle);
            return tuner.getFrameSpec(kernelBundle);
        }
    };
    template<
        typename T_Platform, // <--- Declare T_Platform properly here!
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelBundle>
    struct AdjustThreadSpec::
        Op<cpu::Device<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle,trait::useTuner_v>
    {
        auto operator()(
               cpu::Device<T_Platform> const& device,
               T_Mapping const& executor,
               FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
               T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
        {
            auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
            auto &tuner=Tuner<>::getInstance();
            tuner.disableThreadTuning(kernelBundle);
            auto threadSpec=tuner.tune(device,executor,dataBlocking,kernelBundle);
            return ThreadSpec{threadSpec.m_numBlocks, T_NumThreads::template all<1u>()};
        }

        auto operator()(
            cpu::Device<T_Platform> const& device,
            T_Mapping const& executor,
            FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelBundle const& kernelBundle) const
        {
            std::cout<<"calle neu"<<std::endl;
            auto &tuner=Tuner<>::getInstance();
            tuner.disableThreadTuning(kernelBundle);
            auto threadSpec=tuner.tune(device,executor,dataBlocking,kernelBundle);
            //@TODO add specialization for number of blocks beeing to high
            auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            return ThreadSpec{threadSpec.m_numBlocks, numThreads};
            std::cout<<" I BE EXECUTEd"<<std::endl;
        }
    };
    //ompThreadsAndBlocks
    template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
         struct AdjustThreadSpec::Op<
             cpu::Device<T_Platform>,
             exec::CpuOmpBlocksAndThreads,
             FrameSpec<T_NumBlocks, T_NumThreads>,
             T_KernelBundle,
    trait::useTuner_v>
    {
        auto operator()(
            cpu::Device<T_Platform> const& device,
            exec::CpuOmpBlocksAndThreads const& executor,
            FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
        {
            auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
            auto &tuner=Tuner<>::getInstance();
            tuner.disableThreadTuning(kernelBundle);
            auto threadSpec=tuner.tune(device,executor,dataBlocking,kernelBundle);
            return ThreadSpec{threadSpec.m_numBlocks, T_NumThreads::template all<1u>()};
        }

        auto operator()(
            cpu::Device<T_Platform> const& device,
            exec::CpuOmpBlocksAndThreads const& executor,
            FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelBundle const& kernelBundle) const
        {
            //@TODO handle case where number of threads is > 4 to high
            auto threadSpec=alpaka::tuneWithContext(device,executor,dataBlocking,kernelBundle);
            return threadSpec;
        };
    };
}

#endif //TUNERCPU_HPP
