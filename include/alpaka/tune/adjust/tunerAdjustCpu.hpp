//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include "alpaka/api/cpu/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"

#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/tune/IO/storageTypes.hpp>
#include <alpaka/tune/adjust/adjust.hpp>

namespace alpaka::tune
{
    inline auto primeFactorize(std::size_t max)
    {
        std::size_t start = 2;
        std::vector<std::size_t> factors;
        while(start * start <= max)
        {
            if(max % start == 0)
            {
                factors.push_back(start);
                max /= start;
            }
            else
            {
                start++;
            }
        }
        if(max > 1)
        {
            factors.push_back(max);
        }
        std::sort(factors.rbegin(), factors.rend()); // sort descending
        return factors;
    }

    /*
     * this implements a partition method equally distributing prime factors across dims.
     * this guarentees that for vec.product() is exactly equal to max
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec primeFactorPartitioning(std::size_t max, T_vec const&)
    {
        using ValType = typename T_vec::type;
        auto vecFactors = primeFactorize(max);
        std::vector<ValType> distribute(T_vec::dim(), ValType(1));
        for(auto factor : vecFactors)
        {
            auto min_elem = std::min_element(distribute.begin(), distribute.end());
            *min_elem *= ValType(factor);
        }
        std::sort(distribute.begin(), distribute.end()); // sort ascending (since vec[0] is the slowest index)
        auto resultVec = Vec<ValType, T_vec::dim()>::all(1);
        for(std::size_t i = 0; i < T_vec::dim(); ++i)
        {
            resultVec[i] = distribute[i];
        }
        return resultVec;
    }

    /*
     * this implements a partition method where we take the ceiling of the nth root of the max for each dimension
     * this is a good strategy to distribute work equally does a lot of times more workers are used then necessary
     * very bad for distribution of the threadBlockSize, there primeFactorPartition should be used
     * T
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec ceilRootOverDimPartitioning(std::size_t max, T_vec const&)
    {
        using ValType = typename T_vec::type;
        // start with the 1s Vector
        auto resultVec = Vec<ValType, T_vec::dim()>::all(1);
        auto remainder = max;
        for(std::size_t i = 0; i < T_vec::dim(); ++i)
        {
            // using ceil-root heuristic to distribute mps across dimensions
            ValType split = std::max(ValType(1), static_cast<ValType>(std::pow(remainder, 1.0 / (T_vec::dim() - i))));
            resultVec = split;
            remainder /= split;
        }
        return resultVec;
    }

    /*
     * given a smaller ndim vector returns the the largest multiple of that ndim such that vec.product()<=max
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec multipleOfPartitioning(std::size_t max, T_vec vec)
    {
        using ValType = typename T_vec::type;
        // start with the 1s Vector
        auto resultVec = vec.toRT();
        auto initVec = resultVec;
        while(resultVec.product() <= max - initVec.product())
        {
            // round robin approach of incrementing dims since all of those combinations can be used in the backend

            resultVec += initVec;
        }
        return resultVec;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t ceilRootOverDimPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t primeFactorPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t multipleOfPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

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
        T_KernelRun const& run)
    { // always apply current frameTuning
        auto newRun = makeActiveKernel(
            run.userDefTuneables,
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
                auto stride = alpaka::divCeil(frameSpec.m_numFrames, resultVec);
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
                auto stride = alpaka::divCeil(frameSpec.m_frameExtent, resultVec);
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

    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpec, typename T_KernelRun>
        struct Op
        {
            auto operator()(
                T_Device& device,
                T_Exec const& exec,
                T_FrameSpec const& dataBlocking,
                T_KernelRun const& kernelRun)
            {
                std::cout << " Device: " << typeid(T_Device).name() << std::endl;
                std::cout << " Device: " << alpaka::core::demangledName<T_Device>(device) << std::endl;
                std::cout << "Exec: " << alpaka::core::demangledName<T_Exec>(exec) << std::endl;
                // we can not modify kernelRun or dataBlocking since we need to change their signature
                return std::make_pair(dataBlocking, kernelRun);
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
            T_KernelRun const& kernelRun)
        {
            auto newRun = makeActiveKernel(
                kernelRun.userDefTuneables,
                kernelRun.getNumFramesTune(),
                kernelRun.getFrameExtentTune());
            auto numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            auto numBlocks = Vec<typename T_NumBlocks::type, T_NumBlocks::dim()>::all(1);
            return std::make_pair(alpaka::onHost::ThreadSpec{numBlocks, numThreads}, newRun);
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
            T_KernelRun const& kernelRun)
        {
            //@TODO add specialization
            auto newRun = makeActiveKernel(
                kernelRun.userDefTuneables,
                kernelRun.getNumFramesTune(),
                kernelRun.getFrameExtentTune(),
                kernelRun.getNumBlocksTune());
            auto numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
            if constexpr(newRun.hasNumBlocksTune())
            {
                if(!newRun.getNumBlocksTune().userDef)
                {
                    newRun.getNumBlocksTune().idxRange.m_begin
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getNumBlocksTune().idxRange.m_end = ceilRootOverDimPartitioning(
                        alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount,
                        T_NumThreads{});
                    newRun.getNumBlocksTune().idxRange.m_stride
                        = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                    newRun.getNumBlocksTune().toRange();
                }
            }

            return std::make_pair(
                alpaka::onHost::ThreadSpec{dataBlocking.getThreadSpec().m_numBlocks, numThreads},
                newRun);
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
            T_KernelRun const& kernelRun)
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
                    newRun.getThreadBlockSizeTune().idxRange.m_end = ceilRootOverDimPartitioning(
                        alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount,
                        T_NumThreads{});
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
                        alpaka::onHost::getDeviceProperties(device).m_multiProcessorCount,
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
        T_KernelRun const& run)
    {
        return tunerAdjust::Op<
            ALPAKA_TYPEOF(deviceHandle),
            ALPAKA_TYPEOF(executor),
            alpaka::onHost::FrameSpec<T_NumBlocks, T_NumThreads>,
            T_KernelRun>{}(deviceHandle, executor, dataBlocking, run);
    }


}; // namespace alpaka::tune
#endif // TUNERCPU_HPP
