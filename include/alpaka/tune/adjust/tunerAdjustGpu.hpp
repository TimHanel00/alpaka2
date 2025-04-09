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
            T_KernelRun const& kernelRun)
        {
            auto newRun = kernelRun;
            if constexpr(newRun.hasFrameExtentTune())
            {
                if(!run.getFrameExtentTune().userDef)
                {
                    newRun.getFrameExtentTune().value = frameSpec.m_frameExtent;
                    auto numBlocks = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
                    auto resultVec = primeFactorPartitioning(
                        alpaka::onHost::getDeviceProperties(device).m_warpSize,
                        numBlocks); //-> 2,5,
                    auto stride = alpaka::divCeil(frameSpec, resultVec);
                    newRun.getFrameExtentTune().idxRange = alpaka::IdxRange(stride, frameSpec.m_frameExtent, stride);
                }
            }
            if(!newRun.getThreadBlockSizeTune().userDef)
            {
                newRun.getThreadBlockSizeTune().idxRange.m_begin
                    = primeFactorPartitioning(alpaka::onHost::getDeviceProperties(device).m_warpSize, T_NumThreads{});
                // if(dataBlocking.m_frameExtent.product()<alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock)
                newRun.getThreadBlockSizeTune().idxRange.m_end = multipleOfPartitioning(
                    alpaka::onHost::getDeviceProperties(device).m_maxThreadsPerBlock,
                    newRun.getThreadBlockSizeTune().idxRange.m_begin);
                newRun.getThreadBlockSizeTune().idxRange.m_stride = newRun.getThreadBlockSizeTune().idxRange.m_begin;
            }


            if constexpr(newRun.hasNumBlocksTune())
            {
                if(!newRun.getNumBlocksTune().userDef)
                {
                    newRun.getNumBlocksTune().value.idxRange.m_begin = primeFactorPartitioning(
                        alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount,
                        T_NumBlocks{});
                    newRun.getNumBlocksTune().value.m_end = dataBlocking.m_numFrames;
                    newRun.getNumBlocksTune().value.m_stride = kernelRun.gridSize.idxRange.m_begin;
                }
            }
            return std::make_pair(dataBlocking.getThreadSpec(), newRun);
        }
    };
}; // namespace alpaka::tune

#endif
#endif // TUNERGPU_H
