//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include <alpaka/api/host/Device.hpp>
#include <alpaka/tune/utils/partitioning.hpp>

namespace alpaka::tune
{

    template<typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    static auto adjustThreadSpec(
        auto& deviceHandle,
        auto const& executor,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
        T_KernelRun& run);
#define NrOfNumFrameConfigs 20
#define NrOfFrameExtentConfigs 20

    template<
        typename T_DeviceHandle,
        typename T_Exec,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelRun>
    auto applyHwConstraints(
        T_DeviceHandle device,
        T_Exec exec,
        onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& frameSpec,
        T_KernelRun& run)
    { // always apply current frameTuning
        auto newRun = makeActiveKernel(
            run.userTuneables,
            run.getNumFramesTune(),
            run.getFrameExtentTune(),
            run.getNumBlocksTune(),
            run.getThreadBlockSizeTune());
        using T_newRunType = ALPAKA_TYPEOF(newRun);
        if constexpr(T_newRunType::hasNumFramesTune())
        {
            if(!run.getNumFramesTune().userDef)
            {
                newRun.getNumFramesTune().value = frameSpec.m_numFrames;
                auto numFramesPartitioned = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
                auto resultVec = primeFactorPartitioning(NrOfNumFrameConfigs, numFramesPartitioned); //->z.B 2,5,
                auto stride = frameSpec.m_numFrames / resultVec;
                newRun.getNumFramesTune().idxRange = alpaka::IdxRange(stride, frameSpec.m_numFrames, stride);
            }
        }
        if constexpr(T_newRunType::hasFrameExtentTune())
        {
            if(!run.getFrameExtentTune().userDef)
            {
                newRun.getFrameExtentTune().value = frameSpec.m_frameExtent;
                auto numFramesExtentPartitioned = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                auto resultVec = primeFactorPartitioning(NrOfFrameExtentConfigs, numFramesExtentPartitioned); //-> 2,5,
                auto stride = frameSpec.m_frameExtent / resultVec;
                newRun.getFrameExtentTune().idxRange = alpaka::IdxRange(stride, frameSpec.m_frameExtent, stride);
            }
        }


        auto ret = adjustThreadSpec(device, exec, frameSpec, newRun);
        auto spec = ret.first;
        auto kernel = ret.second;

        return std::make_pair(
            onHost::FrameSpec<T_NumBlocks, T_NumThreads>(
                frameSpec.m_numFrames,
                frameSpec.m_frameExtent,
                spec.m_numBlocks,
                spec.m_numThreads),
            kernel);
    }

    // this is the default tunerAdjust
    //-> it is currenlty designed to fail by default to prevent the compiler from picking no specialization
    //  if you want to prevent this behaviour for a
    //  not yet implemented backend specializtation simply remove the static asserts on the top of the class
    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpec, typename T_KernelRun>
        struct Op
        {
            static_assert(
                !std::is_same_v<T_Device, T_Device>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_Exec, T_Exec>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_FrameSpec, T_FrameSpec>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");

            auto operator()(
                T_Device& device,
                T_Exec const& exec,
                T_FrameSpec const& dataBlocking,
                T_KernelRun& kernelRun)
            {
                std::cout << " Device: " << typeid(T_Device).name() << std::endl;
                std::cout << " Device: " << alpaka::core::demangledName<T_Device>(device) << std::endl;
                std::cout << "Exec: " << alpaka::core::demangledName<T_Exec>(exec) << std::endl;
                // we can not modify kernelRun or dataBlocking since we need to change their signature
                return std::make_pair(dataBlocking.getThreadSpec, kernelRun);
            }
        };
    };

    // serial
    template<typename T_Platform, typename T_Kind, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<T_Platform, T_Kind>,
        alpaka::exec::CpuSerial,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuSerial const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            std::cout << " successfully  found trait spec for cpuSerial: " << core::demangledName<T_Platform>()
                      << std::endl;
            auto newRun = makeActiveKernel(
                kernelRun.userTuneables,
                kernelRun.getNumFramesTune(),
                kernelRun.getFrameExtentTune());
            auto numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            auto numBlocks = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
            return std::make_pair(alpaka::onHost::ThreadSpec{numBlocks, numThreads}, newRun);
        }
    };

    // ompBlocks
    template<typename T_Platform, typename T_Kind, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<T_Platform, T_Kind>,
        alpaka::exec::CpuOmpBlocks,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuOmpBlocks const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            std::cout << " successfully  found trait spec for cpuOmpBlocks: " << core::demangledName<T_Platform>()
                      << std::endl;
            //@TODO add specialization
            auto newRun = makeActiveKernel(
                kernelRun.userTuneables,
                kernelRun.getNumFramesTune(),
                kernelRun.getFrameExtentTune(),
                kernelRun.getNumBlocksTune());
            auto numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            if constexpr(newRun.hasNumBlocksTune())
            {
                if(!newRun.getNumBlocksTune().userDef)
                {
                    newRun.getNumBlocksTune().idxRange.m_begin
                        = primeFactorPartitioning(device.getDeviceProperties().m_multiProcessorCount, T_NumThreads{});
                    newRun.getNumBlocksTune().idxRange.m_end = primeFactorPartitioning(
                        device.getDeviceProperties().m_multiProcessorCount * 4u,
                        T_NumThreads{});
                    newRun.getNumBlocksTune().idxRange.m_stride = primeFactorPartitioning(
                        device.getDeviceProperties().m_multiProcessorCount / 2,
                        T_NumThreads{});
                    newRun.getNumBlocksTune().toRange();
                }
            }

            return std::make_pair(
                alpaka::onHost::ThreadSpec{dataBlocking.getThreadSpec().m_numBlocks, numThreads},
                newRun);
        }
    };

    // ompBlocksAndThreads
    template<typename T_Platform, typename T_Kind, typename T_NumBlocks, typename T_NumThreads, typename T_KernelRun>
    struct tunerAdjust::Op<
        alpaka::onHost::Device<T_Platform, T_Kind>,
        exec::CpuOmpBlocksAndThreads,
        alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
        T_KernelRun>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>& device,
            exec::CpuOmpBlocksAndThreads const& executor,
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
            T_KernelRun& kernelRun)
        {
            auto newRun = kernelRun;
            //@TODO add specialization
            using T_newRunType = ALPAKA_TYPEOF(newRun);
            if constexpr(T_newRunType::hasThreadBlockSizeTune())
            {
                if(!newRun.getThreadBlockSizeTune().userDef)
                {
                    newRun.getThreadBlockSizeTune().idxRange.m_begin
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getThreadBlockSizeTune().idxRange.m_end
                        = primeFactorPartitioning(device.getDeviceProperties().m_multiProcessorCount, T_NumThreads{});
                    newRun.getThreadBlockSizeTune().idxRange.m_stride
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getThreadBlockSizeTune().toRange();
                }
            }
            if constexpr(T_newRunType::hasNumBlocksTune())
            {
                if(!newRun.getNumBlocksTune().userDef)
                {
                    newRun.getNumBlocksTune().idxRange.m_begin
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getNumBlocksTune().idxRange.m_end = ceilRootOverDimPartitioning(
                        device.getDeviceProperties().m_multiProcessorCount,
                        T_NumThreads{});
                    newRun.getNumBlocksTune().idxRange.m_stride
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getNumBlocksTune().toRange();
                }
            }
            return std::make_pair(dataBlocking.getThreadSpec(), newRun);
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
