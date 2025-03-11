//
// Created by tim on 05.03.25.
//

#ifndef TUNERGPU_H
#define TUNERGPU_H
#include "alpaka/api/unifiedCudaHip/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/tune/tunerCpu.hpp"
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
    alpaka::onHost::unifiedCudaHip::Device<alpaka::onHost::unifiedCudaHip::Device<T_Platform>>,
    T_Mapping,
    alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
    T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::unifiedCudaHip::Device<alpaka::onHost::unifiedCudaHip::Device<T_Platform>> & device,
            T_Mapping const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,T_KernelRun &kernelRun)
        {
            using VecType=alpaka::Vec<std::size_t, 1>;
            using idxRangeG=IdxRange<VecType,VecType,VecType>;
            if(kernelRun.threadBlockSize)
            {
                if(!kernelRun.threadBlockSize->userDef)
                {
                    kernelRun.threadBlockSize=alpaka::tune::GridSizeTune{VecType(device->m_properties.m_maxThreadsPerBlock).x()/2,idxRangeG{32,device->m_properties.m_maxThreadsPerBlock,32}};
                }
            }
            if(kernelRun.gridSize)
            {
                if(!kernelRun.gridSize->userDef)
                {
                    kernelRun.gridSize=alpaka::tune::GridSizeTune{VecType(device->m_properties.m_multiProcessorCount).x(),idxRangeG{device->m_properties.m_multiProcessorCount,device->m_properties.m_multiProcessorCount*16,device->m_properties.m_multiProcessorCount}};
                }
            }
            return dataBlocking.getThreadSpec();
        }
    };
};

#endif
#endif //TUNERGPU_H
