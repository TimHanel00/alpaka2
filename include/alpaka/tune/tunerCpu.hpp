//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include "alpaka/api/cpu/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"

namespace alpaka::tune
{

    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpec, typename T_KernelRun>
        struct Op
        {
            auto operator()(
                T_Device& device,
                T_Exec const& exec,
                T_FrameSpec const& dataBlocking,
                T_KernelRun& kernelRun)
            {
                std::cout << " Device: " << typeid(T_Device).name() << std::endl;
                std::cout << " Device: " << alpaka::core::demangledName<T_Device>(device) << std::endl;
                std::cout << "Exec: " << alpaka::core::demangledName<T_Exec>(exec) << std::endl;
                return dataBlocking.getThreadSpec();
            }
        };
    };

    // serial
    template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>,
        alpaka::exec::CpuSerial,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuSerial const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            using VecType = alpaka::Vec<std::size_t, 1>;
            using idxRangeG = IdxRange<VecType, VecType, VecType>;
            //@TODO add specialization
            if(kernelRun.threadBlockSize)
            {
                kernelRun.threadBlockSize = std::nullopt;
            }
            if(kernelRun.gridSize)
            {
                kernelRun.gridSize = std::nullopt;
            }
            auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            auto const numBlocks = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
            return alpaka::onHost::ThreadSpec{numBlocks, numThreads};
        }
    };

    // ompBlocks
    template<
        typename T_Platform,
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>,
        T_Mapping,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            T_Mapping const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            using VecType = alpaka::Vec<std::size_t, 1>;
            using idxRangeG = IdxRange<VecType, VecType, VecType>;
            //@TODO add specialization
            if(kernelRun.threadBlockSize)
            {
                kernelRun.threadBlockSize = std::nullopt;
            }
            if(kernelRun.gridSize)
            {
                if(!kernelRun.gridSize->userDef)
                {
                    using begin = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_begin);
                    kernelRun.gridSize->idxRange.m_begin = begin(1);
                    using end = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_end);
                    kernelRun.gridSize->idxRange.m_end
                        = end(alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount);
                    using stride = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_stride);
                    kernelRun.gridSize->idxRange.m_stride = stride(1);
                }
            }
            auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            return alpaka::onHost::ThreadSpec{dataBlocking.m_threadSpec.m_numBlocks, numThreads};
        }
    };

    // ompBlocksAndThreads
    template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>,
        exec::CpuOmpBlocksAndThreads,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>& device,
            exec::CpuOmpBlocksAndThreads const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            using VecType = alpaka::Vec<std::size_t, 1>;
            using idxRangeG = IdxRange<VecType, VecType, VecType>;
            //@TODO add specialization
            if(kernelRun.threadBlockSize)
            {
                if(!kernelRun.threadBlockSize->userDef)
                {
                    using begin = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_begin);
                    kernelRun.threadBlockSize->idxRange.m_begin = begin(1);
                    using end = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_end);
                    kernelRun.threadBlockSize->idxRange.m_end
                        = end(alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount);
                    using stride = ALPAKA_TYPEOF(kernelRun.threadBlockSize->idxRange.m_stride);
                    kernelRun.threadBlockSize->idxRange.m_stride = stride(1);
                }
            }
            if(kernelRun.gridSize)
            {
                if(!kernelRun.gridSize->userDef)
                {
                    using begin = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_begin);
                    kernelRun.gridSize->idxRange.m_begin = begin(1);
                    using end = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_end);
                    kernelRun.gridSize->idxRange.m_end
                        = end(alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount);
                    using stride = ALPAKA_TYPEOF(kernelRun.gridSize->idxRange.m_stride);
                    kernelRun.gridSize->idxRange.m_stride = stride(1);
                }
            }
            return dataBlocking.getThreadSpec();
        }
    };

    template<typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    static auto adjustThreadSpec(
        auto& deviceHandle,
        auto const& executor,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
        T_KernelRun& run)
    {
        return tunerAdjust::Op<
            ALPAKA_TYPEOF(deviceHandle),
            ALPAKA_TYPEOF(executor),
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
            T_KernelRun>{}(deviceHandle, executor, dataBlocking, run);
    }
}; // namespace alpaka::tune
#endif // TUNERCPU_HPP
