//
// Created by tim on 05.03.25.
//

#ifndef TUNERGPU_H
#define TUNERGPU_H
#include "alpaka/api/unifiedCudaHip/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/adjust/tunerAdjustCpu.hpp>
// #define ALPAKA_LANG_CUDA 1
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
namespace alpaka::tune
{
    template<
        typename T_Platform,
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<alpaka::onHost::unifiedCudaHip::Device<T_Platform>>,
        T_Mapping,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<alpaka::onHost::unifiedCudaHip::Device<T_Platform>>& device,
            T_Mapping const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            if(!kernelRun.threadBlockSize->userDef)
            {
                kernelRun.threadBlockSize->idxRange.m_begin
                    = primeFactorPartitioning(alpaka::onHost::getDeviceProperties(device).m_warpSize, T_NumThreads{});
                // if(dataBlocking.m_frameExtent.product()<alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock)
                kernelRun.threadBlockSize->idxRange.m_end = multipleOfPartitioning(
                    alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock,
                    kernelRun.threadBlockSize->idxRange.m_begin);
                kernelRun.threadBlockSize->idxRange.m_stride = kernelRun.threadBlockSize->idxRange.m_begin;
            }


            if(!kernelRun.gridSize->userDef)
            {
                kernelRun.gridSize->idxRange.m_begin = primeFactorPartitioning(
                    alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount,
                    T_NumBlocks{});
                kernelRun.gridSize->idxRange.m_end = dataBlocking.m_numFrames;
                kernelRun.gridSize->idxRange.m_stride = kernelRun.gridSize->idxRange.m_begin;
            }
            return dataBlocking.getThreadSpec();
        }
    };
}; // namespace alpaka::tune

#endif
#endif // TUNERGPU_H
