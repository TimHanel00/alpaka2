//
// Created by tim on 03.07.25.
//

#ifndef VECTORCTRAIT_H
#define VECTORCTRAIT_H
#include <alpaka/alpaka.hpp>
#include <cstdint>
template<typename CVecTune>
struct VectorAddKernel;
namespace alpaka::tune::trait{

    template<typename CVecTune>
    struct CompileTimeTuneableTrait<VectorAddKernel<CVecTune>>
    {
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1 = tune::CTunable<static_cast<std::size_t>(0),
                CVec<std::uint32_t, 2>,
                CVec<std::uint32_t, 16>,
                CVec<std::uint32_t, 2>>{};
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1}; // empty tuple, no tunables
        }
    };
}
#endif //VECTORCTRAIT_H
