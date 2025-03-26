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
            std::cout << " selected correct " << std::endl;

            if(kernelRun.threadBlockSize)
            {
                if(!kernelRun.threadBlockSize->userDef)
                {
                    auto maxThreads = alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock;
                    using begin = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_begin);
                    kernelRun.threadBlockSize->idxRange.m_begin = begin(32);
                    using end = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_end);
                    kernelRun.threadBlockSize->idxRange.m_end = end(maxThreads);
                    using stride = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_stride);
                    kernelRun.threadBlockSize->idxRange.m_stride = stride(32);
                    kernelRun.threadBlockSize->toRange();
                }
            }
            if(kernelRun.gridSize)
            {
                if(!kernelRun.gridSize->userDef)
                {
                    using begin = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_begin);
                    kernelRun.gridSize->idxRange.m_begin
                        = begin(alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount);
                    using end = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_end);
                    kernelRun.gridSize->idxRange.m_end = end(dataBlocking.m_numFrames.product());
                    using stride = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_stride);
                    kernelRun.gridSize->idxRange.m_stride
                        = stride(alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount);
                    kernelRun.gridSize->toRange();
                }
            }
            return dataBlocking.getThreadSpec();
        }
    };
}; // namespace alpaka::tune

#endif
#endif // TUNERGPU_H
