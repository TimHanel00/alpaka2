//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include "alpaka/api/cpu/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"

namespace alpaka::tune{

    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpec, typename T_KernelRun>
    struct Op {
            auto operator()(T_Device &,
                            T_Exec const & exec,
                            T_FrameSpec const & dataBlocking,
                            T_KernelRun & kernelRun)
            {
                return dataBlocking.getThreadSpec();
            }
        };
    };
    //serial
    template<
        typename T_Platform,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelRun>
        struct tunerAdjust::Op<
        alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>,
        alpaka::exec::CpuSerial,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>> & device,//@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuSerial const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun & kernelRun)
        {
            std::cout<<" CORRECT "<<std::endl;
            using VecType=alpaka::Vec<std::size_t, 1>;
            using idxRangeG=IdxRange<VecType,VecType,VecType>;
            //@TODO add specialization
            if(kernelRun.threadBlockSize)
            {
                kernelRun.threadBlockSize=std::nullopt;
            }
            if(kernelRun.gridSize)
            {
                kernelRun.gridSize=std::nullopt;
            }
            auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            auto const numBlocks = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
            return alpaka::onHost::ThreadSpec{numBlocks, numThreads};
        }
    };
    //ompBlocks
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
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>> & device,//@TODO fix this its a bug with that extra wrapped layer
            T_Mapping const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun & kernelRun)
        {
                using VecType=alpaka::Vec<std::size_t, 1>;
                using idxRangeG=IdxRange<VecType,VecType,VecType>;
                //@TODO add specialization
                if(kernelRun.threadBlockSize)
                {
                    kernelRun.threadBlockSize=std::nullopt;
                }
                if(kernelRun.gridSize)
                {
                    if(!kernelRun.gridSize->userDef)
                    {
                        kernelRun.gridSize=alpaka::tune::GridSizeTune{VecType(device->m_properties.m_multiProcessorCount).x(),idxRangeG{1,device->m_properties.m_multiProcessorCount,1}};
                    }
                }
                auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                return alpaka::onHost::ThreadSpec{dataBlocking.m_threadSpec.m_numBlocks, numThreads};
            }
        };
        //ompBlocksAndThreads
        template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
             struct tunerAdjust::Op<
                 alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>>,
                 exec::CpuOmpBlocksAndThreads,
                 alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
                 T_KernelRun>
        {
            auto operator()(
            alpaka::onHost::Device<alpaka::onHost::cpu::Device<T_Platform>> & device,
            exec::CpuOmpBlocksAndThreads const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun & kernelRun)
            {
                using VecType=alpaka::Vec<std::size_t, 1>;
                using idxRangeG=IdxRange<VecType,VecType,VecType>;
                //@TODO add specialization
                if(kernelRun.threadBlockSize)
                {
                    if(!kernelRun.threadBlockSize->userDef)
                    {
                        kernelRun.threadBlockSize=alpaka::tune::ThreadBlockSizeTune{VecType(device->m_properties.m_multiProcessorCount).x(),idxRangeG{1,device->m_properties.m_multiProcessorCount,1}};
                    }
                }
                if(kernelRun.gridSize)
                {
                    if(!kernelRun.gridSize->userDef)
                    {
                        kernelRun.gridSize=alpaka::tune::GridSizeTune{VecType(device->m_properties.m_multiProcessorCount).x(),idxRangeG{1,device->m_properties.m_multiProcessorCount,1}};
                    }
                }
                return dataBlocking.getThreadSpec();
            }
        };
    template<typename T_NumBlocks, typename T_NumThreads,typename T_KernelRun>
    static auto adjustThreadSpec(
        auto & deviceHandle,
        auto const& executor,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
        T_KernelRun &run)
    {
        return tunerAdjust::Op<
            ALPAKA_TYPEOF(deviceHandle),
            ALPAKA_TYPEOF(executor),
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
            T_KernelRun>{}(deviceHandle, executor, dataBlocking,run);

    }
};
#endif //TUNERCPU_HPP
