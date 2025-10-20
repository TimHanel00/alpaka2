//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include "alpaka/tune/tunable/frameSpecTuningModel.hpp"

#include <alpaka/api/host/Device.hpp>
#include <alpaka/tune/utils/partitioning.hpp>

namespace alpaka::tune::adjust
{
#define NrOfNumFrameConfigs 16
#define NrOfFrameExtentConfigs 16
#define defaultMinSteps 8
#define defaultMaxSteps 16

    /**
     * @brief Generate candidate vectors by expanding a per-dimension partition up to a maximum.
     *
     * For each dimension, computes a step size as an integer multiple of the provided
     * @p partitionedVec such that at least @p minSteps (and at most @p maxSteps) steps fit
     * within @p maxVal. Starts from the computed step and repeatedly adds the step until
     * the next value would reach or exceed @p maxVal. Returns all generated vectors.
     *
     * Edge handling:
     * - If @p minSteps == 0 or @p maxSteps == 0, returns an empty list.
     * - Zero entries in @p partitionedVec are treated as 1.
     * - If the initial step would yield fewer than @p minSteps, it is increased (in multiples
     *   of the base partition) to satisfy the minimum, clamped below @p maxVal.
     *
     * @tparam VecType  Alpaka vector type. Must obey Vector concept (alpaka::concept::Vector).
     *
     * @param maxVal          Per-dimension exclusive upper bounds.
     * @param partitionedVec  Per-dimension base partition (step granularity).
     * @param minSteps        Minimum number of steps desired per dimension (default: implementation constant).
     * @param maxSteps        Upper bound on number of candidates to emit (default: implementation constant).
     *
     * @return std::vector<VecType>  Sequence of candidate vectors in ascending order,
     *                               each strictly less than @p maxVal component-wise.
     *
     * @pre  @p maxVal and @p partitionedVec have the same dimensionality.
     * @complexity O(D + K), where D is dimensionality and K ≤ @p maxSteps is the number of emitted candidates.
     */
    auto boundedPartitionExpansion(
        alpaka::concepts::Vector auto const& maxVal,
        alpaka::concepts::Vector auto const& partitionedVec,
        std::size_t minSteps = defaultMinSteps,
        std::size_t maxSteps = defaultMaxSteps)
    {
        using Vec = std::remove_cvref_t<decltype(maxVal)>;
        using Scalar = typename Vec::type;

#ifdef Debug
        std::cout << "[TuneStep] minSteps = " << minSteps << ", maxSteps = " << maxSteps << "\n";
        std::cout << "[TuneStep] maxVal = " << maxVal << ", partitionedVec = " << partitionedVec << "\n";
#endif

        // Early exit if no work
        if(minSteps == 0 || maxSteps == 0)
        {
#ifdef Debug
            std::cout << "[EarlyExit] Skipping due to minSteps or maxSteps being 0.\n";
#endif
            return std::vector<Vec>{};
        }

        Vec baseStep = partitionedVec;
        Vec step{};

#ifdef Debug
        std::cout << "[StepCalc] Starting step computation loop...\n";
#endif

        for(std::size_t i = 0; i < alpaka::getDim(step); ++i)
        {
            Scalar base = baseStep[i];
            if(base == 0)
            {
#ifdef Debug
                std::cerr << "[Warning] baseStep[" << i << "] = 0, forcing to 1.\n";
#endif
                base = 1;
            }

            Scalar rawStep = maxVal[i] / static_cast<Scalar>(maxSteps);
            double divDown = static_cast<double>(rawStep) / static_cast<double>(base);
            Scalar nDown = static_cast<Scalar>(std::floor(divDown));
            Scalar candidate = std::max(nDown * base, base);
            step[i] = candidate;

#ifdef Debug
            std::cout << "[StepCalc] Dim " << i << ":\n";
            std::cout << "  baseStep = " << baseStep[i] << ", maxVal = " << maxVal[i] << "\n";
            std::cout << "  rawStep = " << rawStep << ", divDown = " << divDown << ", nDown = " << nDown << "\n";
            std::cout << "  Initial step = " << step[i] << "\n";
#endif

            Scalar numSteps = maxVal[i] / step[i];
#ifdef Debug
            std::cout << "  numSteps = " << numSteps << " (vs minSteps = " << minSteps << ")\n";
#endif

            if(numSteps < minSteps)
            {
                Scalar minStep = maxVal[i] / static_cast<Scalar>(minSteps);
                double divUp = static_cast<double>(minStep) / static_cast<double>(base);
                auto nUp = static_cast<Scalar>(std::ceil(divUp));
                Scalar adjusted = nUp * base;
                step[i] = std::max(adjusted, Scalar(1));

#ifdef Debug
                std::cout << "  [Fallback] minStep = " << minStep << ", divUp = " << divUp << ", nUp = " << nUp
                          << ", adjusted = " << adjusted << ", final fallback step = " << step[i] << "\n";
#endif
                if(adjusted >= maxVal[i])
                {
                    step[i] = std::max(minStep, Scalar(1));
#ifdef Debug
                    std::cout << "  [Adjusted] Step too large. Using minStep fallback: " << step[i] << "\n";
#endif
                }
            }

            if(step[i] == 0)
            {
#ifdef Debug
                std::cerr << "[Error] Final step[" << i << "] is 0! Forcing to 1.\n";
#endif
                step[i] = 1;
            }
        }

        std::vector<Vec> values;
        Vec current = step;

#ifdef Debug
        std::cout << "[ValueGen] Generating values starting from step: " << step << "\n";
#endif

        while(allTrue(current < maxVal) && values.size() < maxSteps)
        {
#ifdef Debug
            std::cout << "  Adding value: " << current << "\n";
#endif
            values.emplace_back(current);
            current = current + step;
        }

#ifdef Debug
        std::cout << "[Result] Generated " << values.size() << " values:\n";
        std::ranges::for_each(values, [](auto const& v) { std::cout << v << ' '; });
        std::cout << '\n';
#endif

        return values;
    }

    /**
     * @brief Replace shallow placeholders (user-unspecified spaces) with tuner-chosen spaces of numFrames and
     * frameExtent tune.
     *
     * Replacement happens with recursion.
     * If no branch is selected, we return the current model unchanged—meaning all shallowTunables have been
     * replaced already.
     *
     * @tparam T_DeviceHandle Device handle type.
     * @tparam T_Exec         Executor/mapping type.
     * @tparam T_FrameSpecTuningModel FrameSpec tuning model (may contain shallow tunables).
     * @param device          Target device whose properties guide space derivation.
     * @param exec            Executor/mapping context (not modified).
     * @param frameSpecTune   Current (possibly partially shallow) tuning model.
     * @return A FrameSpec tuning model with the next shallow placeholder replaced, or the original model if none
     * remain.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrames(T_DeviceHandle, T_Exec, T_FrameSpecTuningModel&&);

    namespace detail
    {
        /** @brief Resolve a shallow *numFrames* placeholder (reuse numBlocks or derive via partitioning) and recurse.
         *  @param device Device used to derive space; @param exec Executor; @param fs current model.
         *  @return Model with *numFrames* made concrete (or passthrough if none). */
        template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTune>
        auto adjustNumFrames(T_DeviceHandle device, T_Exec exec, T_FrameSpecTune&& fs)
        {
            using FSTM = std::remove_cvref_t<T_FrameSpecTune>;
            using Spec = std::remove_cvref_t<decltype(fs.m_spec)>;
            if constexpr(FSTM::hasNumBlocksTune())
            {
                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumBlocksTune(),
                        fs.getFrameExtentTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
            else
            {
                auto base = Spec::NumFramesVecType::all(1);
                auto factors = primeFactorPartitioning(NrOfNumFrameConfigs, base);
                auto stride = fs.m_spec.m_numFrames / factors;
                auto tune = TunableMD<tune::frame::numFrames>{alpaka::IdxRange(stride, fs.m_spec.m_numFrames, stride)};

                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        tune,
                        fs.getFrameExtentTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
        }

        /** @brief Resolve a shallow *frameExtent* placeholder (reuse numThreads or derive via partitioning) and
         * recurse.
         *  @param device Device used to derive space; @param exec Executor; @param fs current model.
         *  @return Model with *frameExtent* made concrete (or passthrough if none). */
        template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTune>
        auto adjustFrameExtent(T_DeviceHandle device, T_Exec exec, T_FrameSpecTune&& fs)
        {
            using FSTM = std::remove_cvref_t<T_FrameSpecTune>;
            using Spec = std::remove_cvref_t<decltype(fs.m_spec)>;

            if constexpr(FSTM::hasNumThreadsTune())
            {
                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumFramesTune(),
                        fs.getNumThreadsTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
            else
            {
                auto base = Spec::FrameExtentsVecType::all(1);
                auto factors = primeFactorPartitioning(NrOfFrameExtentConfigs, base);
                auto stride = fs.m_spec.m_frameExtent / factors;
                auto tune
                    = TunableMD<tune::frame::frameExtent>{alpaka::IdxRange(stride, fs.m_spec.m_numFrames, stride)};

                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumFramesTune(),
                        tune,
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
        }
    } // namespace detail

    /**
     * @brief Replace shallow placeholders (user-unspecified spaces) with tuner-chosen spaces of numFrames and
     * frameExtent tune.
     *
     * Replacement happens with recursion.
     * If no branch is selected, we return the current model unchanged—meaning all shallowTunables have been
     * replaced already.
     *
     * @tparam T_DeviceHandle Device handle type.
     * @tparam T_Exec         Executor/mapping type.
     * @tparam T_FrameSpecTuningModel FrameSpec tuning model (may contain shallow tunables).
     * @param device          Target device whose properties guide space derivation.
     * @param exec            Executor/mapping context (not modified).
     * @param frameSpecTune   Current (possibly partially shallow) tuning model.
     * @return A FrameSpec tuning model with the next shallow placeholder replaced, or the original model if none
     * remain.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrames(T_DeviceHandle device, T_Exec exec, T_FrameSpecTuningModel&& frameSpecTune)
    {
        using FSTM = std::remove_cvref_t<T_FrameSpecTuningModel>;

        if constexpr(
            FSTM::hasNumFramesTune()
            && alpaka::tune::concepts::shallowTunable<std::remove_cvref_t<decltype(frameSpecTune.getNumFramesTune())>>)
            return detail::adjustNumFrames(device, exec, std::forward<T_FrameSpecTuningModel>(frameSpecTune));
        else if constexpr(
            FSTM::hasFrameExtentTune()
            && alpaka::tune::concepts::shallowTunable<
                std::remove_cvref_t<decltype(frameSpecTune.getFrameExtentTune())>>)
            return detail::adjustFrameExtent(device, exec, std::forward<T_FrameSpecTuningModel>(frameSpecTune));
        else
            return std::forward<T_FrameSpecTuningModel>(frameSpecTune);
    }

    // forward declare
    static auto adjustThreadSpec(auto& deviceHandle, auto const& executor, auto const& dataBlocking);
    template<typename T_>
    struct DummyV;

    /**
     * @brief Adjusts a FrameSpec tuning model for the given device and executor.
     *
     * Applies hardware constraints (e.g., max threads, blocks) and disables
     * unsupported tunings. If no user-defined tunables exist (frameTuneModel contains shallowTunable), derives a
     * tuning space from device resources (e.g., multiprocessors, warp size) using prime-factor partitioning.
     *
     * @return The adjusted FrameSpec tuning model.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrameSpecTune(T_DeviceHandle device, T_Exec exec, T_FrameSpecTuningModel const& frameSpecTune)
    { // always apply current frameTuning
        // adjust thread spec (numBlock,numThreads)
        auto newSpecTune = adjustThreadSpec(device, exec, frameSpecTune);
        // adjust frames (numFrames,frameExtent)
        return adjustFrames(device, exec, newSpecTune);
    }
    template<typename T>
    struct Dummy;

    // this is the default tunerAdjust
    //-> it is currenlty designed to fail by default to prevent the compiler from picking no specialization
    //  if you want to prevent this behaviour for a
    //  not yet implemented backend specializtation simply remove the static asserts on the top of the class
    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpecTune>
        struct Op
        {
            Dummy<T_Device> dummy;
            Dummy<T_Exec> dummy2;
            Dummy<T_FrameSpecTune> dummy3;
            static_assert(
                !std::is_same_v<T_Device, T_Device>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_Exec, T_Exec>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_FrameSpecTune, T_FrameSpecTune>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");

            auto operator()(T_Device& device, T_Exec const& exec, T_FrameSpecTune&& dataBlocking)
            {
                auto j = DummyV<decltype(dataBlocking)>{};
                // we can not modify kernelRun or dataBlocking since we need to change their signature
                return dataBlocking;
            }
        };
    };

    // serial
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<alpaka::onHost::Device<T_Platform, T_Kind>, alpaka::exec::CpuSerial, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuSerial const& executor,
            T_FrameSpecTuningModel const& frameTuningModel)
        {
            using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
            using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;

            /*
            *        using type = typename T_NumFrames::type;

        using NumFramesVecType = T_NumFrames;
        using FrameExtentsVecType = T_FrameExtents;
        using ThreadExtentsVecType = T_ThreadExtents;
        using ThreadSpecType = ThreadSpec<T_NumFrames, T_ThreadExtents>;

            */
            auto numThreads = numThreads_type::all(1);
            auto numBlocks = numBlocks_type::all(1);

            auto v = FrameSpecTuningModel{
                onHost::FrameSpec{
                    frameTuningModel.m_spec.m_numFrames,
                    frameTuningModel.m_spec.m_frameExtent,
                    numBlocks,
                    numThreads},
                frameTuningModel.getNumFramesTune(),
                frameTuningModel.getFrameExtentTune()};
            return v;
        }
    };

    // ompBlocks
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::
        Op<alpaka::onHost::Device<T_Platform, T_Kind>, alpaka::exec::CpuOmpBlocks, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuOmpBlocks const& executor,
            T_FrameSpecTuningModel&& frameTuningModel)
        {
#ifdef Debug
            std::cout << " successfully  found trait spec for cpuOmpBlocks: " << onHost::demangledName<T_Platform>()
                      << std::endl;
#endif
            //@TODO add specialization

            using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
            using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;
            auto numThreads = numThreads_type::all(1);
            if constexpr(
                T_FrameSpecTuningModel::hasNumBlocksTune()
                && alpaka::tune::concepts::shallowTunable<
                    std::remove_cvref_t<decltype(frameTuningModel.getNumBlocksTune())>>)
            {
                auto partitionedCores
                    = primeFactorPartitioning(device.getDeviceProperties().m_multiProcessorCount, numBlocks_type{});
                auto startVec = partitionedCores;
                auto vec
                    = boundedPartitionExpansion<numBlocks_type>(frameTuningModel.m_spec.m_numFrames, partitionedCores);
                auto tune = TunableMD<tune::frame::numBlocks>{std::move(vec)};
                auto neuSpec = FrameSpecTuningModel{
                    onHost::FrameSpec{
                        frameTuningModel.m_spec.m_numFrames,
                        frameTuningModel.m_spec.m_frameExtent,
                        frameTuningModel.m_spec.m_threadSpec.m_numBlocks,
                        numThreads},
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    std::move(tune)};
                using device_T = decltype(device);
                using executor_T = decltype(executor);
                using frameSpec_T = decltype(neuSpec);
                return tunerAdjust::Op<device_T, executor_T, decltype(neuSpec)>{}(device, executor, neuSpec);
            }
            else
            {
                return FrameSpecTuningModel{
                    onHost::FrameSpec{
                        frameTuningModel.m_spec.m_numFrames,
                        frameTuningModel.m_spec.m_frameExtent,
                        frameTuningModel.m_spec.m_threadSpec.m_numBlocks,
                        numThreads},
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    frameTuningModel.getNumBlocksTune()};
            }
        };
    };

    static auto adjustThreadSpec(auto& deviceHandle, auto const& executor, auto const& specTune)

    {
        using bareDevice = std::remove_cvref_t<decltype(deviceHandle)>;
        using bareExecutor = std::remove_cvref_t<decltype(executor)>;
        using bareTune = std::remove_cvref_t<decltype(specTune)>;
        return tunerAdjust::Op<bareDevice, bareExecutor, bareTune>{}(deviceHandle, executor, specTune);
    }

}; // namespace alpaka::tune::adjust
#endif // TUNERCPU_HPP
