//
// Created by tim on 29.04.25.
//

#ifndef RESIDUALKERNEL_H
#define RESIDUALKERNEL_H
#include <alpaka/alpaka.hpp>
using namespace alpaka;
static bool afterInit = false;

template<typename Vec2>
struct computeResidualReduceKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto pressureField,
        alpaka::concepts::MdSpan auto rhs,
        alpaka::concepts::MdSpan auto residualReduced,
        alpaka::concepts::CVector auto const sharedMemExtents,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::Vector auto const numNodes,

        auto dx,
        auto dy) const -> void
    {
        // Vec2 xDir{0u, 1u};
        // Vec2 yDir{1u, 0u};

        constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
        constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};
        // blockSizeExtent would be a better fit
        for(alpaka::concepts::Dim<2u> auto blockStartIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, chunkSize}))
        {
            auto pressurSdata = alpaka::onAcc::declareSharedMdArray<double, alpaka::uniqueId()>(acc, sharedMemExtents);
            auto residualSdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
            {
                auto bufIdx = idx2d + blockStartIdx;
                pressurSdata[idx2d] = pressureField[bufIdx];
                residualSdata[acc[layer::thread].idx()] = 0.0;
            }

            onAcc::syncBlockThreads(acc);
            // avoid data race with the stencil calculation at the end

            for(alpaka::concepts::Dim<2u> auto idx2d : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInBlock,
                    IdxRange{chunkSize} >> 1u,
                    onAcc::traverse::tiled))
            {
                auto bufIdx = idx2d + blockStartIdx;
                double laplacian
                    = (pressurSdata[idx2d + xDir] - 2.0 * pressurSdata[idx2d] + pressurSdata[idx2d - xDir]) / (dx * dx)
                      + (pressurSdata[idx2d + yDir] - 2.0 * pressurSdata[idx2d] + pressurSdata[idx2d - yDir])
                            / (dy * dy);
                double h = laplacian - rhs[bufIdx];
                residualSdata[acc[layer::thread].idx()] += (h * h);
            }
            onAcc::syncBlockThreads(acc);
            /*
            std::cout << " accumulated: " << residualSdata[acc[layer::thread].idx()] << std::endl;
            for(int i = 0; i < sharedMemExtents[0]; i++)
            {
                for(int j = 0; j < sharedMemExtents[1]; j++)
                {
                    std::cout << "y: " << i << "x: " << j << " data " << residualSdata[Vec<uint32_t, 2>{i, j}] << "\n";
                }
            }*/
            // block reduce for every chunk
            auto local_i = acc[layer::thread].idx(); // y,x
            auto const blockSize = acc[layer::thread].count(); // y,x
            // Phase 1: reduce along Y (all participates)
            for(uint32_t stride = blockSize[0] / 2; stride > 0; stride /= 2)
            {
                onAcc::syncBlockThreads(acc);
                if(local_i[0] < stride)
                {
                    residualSdata[local_i] += residualSdata[Vec2{local_i[0] + stride, local_i[1]}];
                }
            }

            // Phase 2: reduce along X (only row 0 participates)
            for(uint32_t stride = blockSize[1] / 2; stride > 0; stride /= 2)
            {
                onAcc::syncBlockThreads(acc);
                if(local_i[1] < stride && local_i[0] == 0)
                {
                    residualSdata[local_i] += residualSdata[Vec2{0, local_i[1] + stride}];
                }
            }

            if(local_i[0] == 0 && local_i[1] == 0)
            {
                onAcc::atomicAdd(acc, &residualReduced[0], residualSdata[local_i]);
            }
        }


        // auto offset = blockSize / 2; offset > 0; offset /= 2

        /*
        // aggregate for each thread but skip the first shared memory slot
        for(auto elemIdxInFrame : onAcc::makeIdxMap(
                acc,
                onAcc::worker::threadsInBlock,
                IdxRange{acc[layer::thread].count(), chunkSize} >> 1u,
                onAcc::traverse::tiled))
        {
            residualSdata[acc[layer::thread].idx()] += residualSdata[elemIdxInFrame];
        }
*/
    }
};
#endif // RESIDUALKERNEL_H
