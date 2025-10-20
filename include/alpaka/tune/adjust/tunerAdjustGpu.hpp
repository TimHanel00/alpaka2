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
    template<typename T_Platform, typename T_Kind, typename T_Mapping, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<alpaka::onHost::Device<T_Platform, T_Kind>, T_Mapping, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>& device,
            T_Mapping const& executor,
            T_FrameSpecTuningModel&& frameTuningModel)
        {
            using Spec = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using NumBlocks = typename Spec::ThreadSpecType::NumBlocksVecType;
            using NumThreads = typename Spec::ThreadSpecType::NumThreadsVecType;

            // ---------- numThreads tuning (using multipleOfPartitioning) ----------
            if constexpr(
                T_FrameSpecTuningModel::hasNumThreadsTune()
                && alpaka::tune::concepts::shallowTunable<
                    std::remove_cvref_t<decltype(frameTuningModel.getNumThreadsTune())>>)
            {
                auto begin = primeFactorPartitioning(device.getDeviceProperties().m_warpSize, NumThreads{});
                auto end = multipleOfPartitioning(device.getDeviceProperties().m_maxThreadsPerBlock, begin);
                auto stride = begin;

                auto numThreadsTune = TunableMD<tune::frame::numThreads>{alpaka::IdxRange(begin, end, stride)};

                // keep other tunables as-is
                auto neuSpec = FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    frameTuningModel.getNumBlocksTune(),
                    std::move(numThreadsTune)};

                using DevT = decltype(device);
                using ExecT = decltype(executor);
                return tunerAdjust::Op<DevT, ExecT, decltype(neuSpec)>{}(device, executor, neuSpec);
            }

            // ---------- numBlocks tuning (use boundedPartitionExpansion) ----------
            else if constexpr(
                T_FrameSpecTuningModel::hasNumBlocksTune()
                && alpaka::tune::concepts::shallowTunable<
                    std::remove_cvref_t<decltype(frameTuningModel.getNumBlocksTune())>>)
            {
                // seed by multiprocessor count
                auto partitionedMP
                    = primeFactorPartitioning(device.getDeviceProperties().m_multiProcessorCount, NumBlocks{});

                // generate candidate list using your vector-producing helper
                // (min/max steps can be adjusted to match your original 4..8 behaviour)
                auto values = boundedPartitionExpansion<NumBlocks>(
                    frameTuningModel.m_spec.m_numFrames, // max
                    partitionedMP, // partition base
                    /*minSteps*/ 4,
                    /*maxSteps*/ 8);

                auto numBlocksTune = TunableMD<tune::frame::numBlocks>{std::move(values)};

                auto neuSpec = FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    std::move(numBlocksTune),
                    frameTuningModel.getNumThreadsTune()};

                using DevT = decltype(device);
                using ExecT = decltype(executor);
                return tunerAdjust::Op<DevT, ExecT, decltype(neuSpec)>{}(device, executor, neuSpec);
            }
            else
            {
                // ---------- nothing to adjust; pass-through ----------
                return FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    frameTuningModel.getNumBlocksTune(),
                    frameTuningModel.getNumThreadsTune()};
            }
        }
    };
}; // namespace alpaka::tune

#endif
#endif // TUNERGPU_H
