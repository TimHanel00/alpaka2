//
// Created by tim on 29.04.25.
//

#ifndef RESIDUALKERNEL_H
#define RESIDUALKERNEL_H
#include <alpaka/alpaka.hpp>
using namespace alpaka;

struct computeResidualReduceKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto residual,

        alpaka::concepts::MdSpan auto pressureField,
        alpaka::concepts::MdSpan auto rhs,
        alpaka::concepts::MdSpan auto residualReduced,
        alpaka::concepts::CVector auto const sharedMemExtents,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::Vector auto const numNodes,

        auto dx,
        auto dy)
    {
        auto traverseBlock = alpaka::onAcc::makeIdxMap(
            acc,
            alpaka::onAcc::worker::threadsInBlock,
            alpaka::IdxRange{sharedMemExtents});
        constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
        constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};
        auto residualSdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);
        for(alpaka::concepts::Dim<2u> auto blockStartIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, chunkSize}))
        {
            auto pressurSdata = alpaka::onAcc::declareSharedMdArray<double, alpaka::uniqueId()>(acc, sharedMemExtents);


            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
            {
                auto bufIdx = idx2d + blockStartIdx;
                pressurSdata[idx2d] = pressureField[bufIdx];
            }

            onAcc::syncBlockThreads(acc);
            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize} >> 1),
                onAcc::traverse::tiled)
            {
                auto bufIdx = idx2d + blockStartIdx;
                residual[idx2d]
                    = (pressurSdata[idx2d + xDir] - 2. * pressurSdata[idx2d] + pressurSdata[idx2d - 1]) / (dx * dx)
                      + (pressurSdata[idx2d + yDir] - 2. * pressurSdata[idx2d] + pressurSdata[idx2d - yDir])
                            / (dy * dy)
                      - rhs[bufIdx];
                residualSdata[idx2d] = 0.0;
            }

            onAcc::syncBlockThreads(acc);
            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize} >> 1),
                onAcc::traverse::tiled)
            {
                auto bufIdx = idx2d + blockStartIdx;

                auto allThreads = onAcc::SimdAlgo{onAcc::WorkerGroup{bufIdx, numNodes}};
                auto reducedValue = allThreads.transformReduce(
                    acc,
                    alpaka::Vec{numNodes},
                    double_t{0},
                    std::plus{},
                    [&](auto const&, auto&& simdA) constexpr
                    {
                        auto tmp = simdA.load();
                        // tmpValue += tmp.sum();
                        return tmp;
                    },
                    residual);
                residualSdata[idx2d] += reducedValue;
            }
        }
        onAcc::syncBlockThreads(acc);
        // aggregate for each thread but skip the first shared memory slot
        for(auto [elemIdxInFrame] : onAcc::makeIdxMap(
                acc,
                onAcc::worker::threadsInBlock,
                IdxRange{acc[layer::thread].count(), chunkSize} >> 1))
        {
            residualSdata[acc[layer::thread].idx()] += residualSdata[elemIdxInFrame];
        }
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            onAcc::syncBlockThreads(acc);
            if(local_i < offset)
                residualSdata[local_i] += residualSdata[local_i + offset];
        }
        if(local_i == 0)
            onAcc::atomicAdd(acc, &residualReduced[0], residualSdata[local_i]);
    }
};
#endif // RESIDUALKERNEL_H
