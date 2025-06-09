//
// Created by tim on 05.03.25.
//

#ifndef TUNERGPU_H
#define TUNERGPU_H
// #define ALPAKA_LANG_CUDA 1
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP || ALPAKA_LANG_SYCL
#    include <alpaka/api/unifiedCudaHip/Device.hpp>
#    include <alpaka/tune/utils/partitioning.hpp>

namespace alpaka::tune
{
    template<typename T_Platform,typename T_Kind, typename T_Mapping,typename T_NumBlocks,typename T_NumThreads,typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<T_Platform,T_Kind>,
        T_Mapping,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform,T_Kind>& device,
            T_Mapping const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            auto newRun = kernelRun;
            using T_newActiveRunType = ALPAKA_TYPEOF(newRun);
            if constexpr(T_newActiveRunType::hasThreadBlockSizeTune())
            {
                if(!newRun.getThreadBlockSizeTune().userDef)
                {
                    newRun.getThreadBlockSizeTune().idxRange.m_begin = primeFactorPartitioning(
                        device.getDeviceProperties().m_warpSize,
                        T_NumThreads{});
                    // if(dataBlocking.m_frameExtent.product()<alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock)
                    newRun.getThreadBlockSizeTune().idxRange.m_end = multipleOfPartitioning(
                        device.getDeviceProperties().m_maxThreadsPerBlock,
                        newRun.getThreadBlockSizeTune().idxRange.m_begin);
                    newRun.getThreadBlockSizeTune().idxRange.m_stride
                        = newRun.getThreadBlockSizeTune().idxRange.m_begin;
                }
            }

            if constexpr(T_newActiveRunType::hasNumBlocksTune())
            {
                if(!newRun.getNumBlocksTune().userDef)
                {
                    newRun.getNumBlocksTune().idxRange.m_begin = primeFactorPartitioning(
                        device.getDeviceProperties().m_multiProcessorCount * 16u,
                        T_NumBlocks{});
                    newRun.getNumBlocksTune().idxRange.m_end = dataBlocking.m_numFrames;
                    newRun.getNumBlocksTune().idxRange.m_stride = newRun.getNumBlocksTune().idxRange.m_begin;
                }
            }
            return std::make_pair(dataBlocking.getThreadSpec(), newRun);
        }
    };
}; // namespace alpaka::tune

#endif
#endif // TUNERGPU_H
