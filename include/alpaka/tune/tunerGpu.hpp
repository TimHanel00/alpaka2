//
// Created by tim on 05.03.25.
//

#ifndef TUNERGPU_H
#define TUNERGPU_H
#include "alpaka/api/unifiedCudaHip/Device.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/trait.hpp"
#include "tuner.hpp"
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
    #include <alpaka/alpaka.hpp>
    namespace alpaka::onHost::internal
    {
        template<
                           typename T_Platform,
                           typename T_Mapping,
                           typename T_NumBlocks,
                           typename T_NumThreads,
                           typename T_KernelBundle>
                            require trait::useTuner_v)
                       struct AdjustThreadSpec::
                           Op<unifiedCudaHip::Device<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            auto operator()(
                unifiedCudaHip::Device<T_Platform> const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle & kernelBundle) const
            {
                auto threadSpec=alpaka::tuneWithContext(device,executor,dataBlocking,kernelBundle);
                return threadSpec;
            }
        }
    }
#endif
#endif //TUNERGPU_H
