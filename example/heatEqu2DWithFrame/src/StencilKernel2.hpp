/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param sharedMemExtents size of the shared memory box
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t

#include <iostream>
inline int writeLoopAccs = 0;
inline int computeLoopAccs = 0;

void verifyCorrectness(auto& numNodes, auto& frameExtent)
{
    constexpr auto haloSize = 1;
    auto tilesPerX = (numNodes.x() + frameExtent.x() - 1) / frameExtent.x();
    auto tilesPerY = (numNodes.y() + frameExtent.y() - 1) / frameExtent.y();
    auto totalLoads = tilesPerX * tilesPerY * (frameExtent.x() + 2 * haloSize) * (frameExtent.y() + 2 * haloSize);
    std::cout << totalLoads << " vs " << writeLoopAccs << std::endl;
    std::cout << numNodes.product() << " vs " << computeLoopAccs << std::endl;
    if(numNodes.y() % frameExtent.y() != 0)
    {
        std::cout << numNodes.y() << " numNodes y is not divisible by  frameExtent y" << frameExtent.y() << std::endl;
        std::terminate();
    }
    if(numNodes.x() % frameExtent.x() != 0)
    {
        std::cout << numNodes.x() << " numNodes x is not divisible by  frameExtent x" << frameExtent.y() << std::endl;

        /*throw std::runtime_error("frameExtent has to divide numNodes -- correctness constraint violated");*/
        std::terminate();
    }


    if(static_cast<int>(totalLoads) != writeLoopAccs)
    {
        std::string prod = std::to_string(static_cast<int>(totalLoads));
        std::cout << "incorrect number of read accesses to global buffer, expected: " + prod + " vs "
                         + std::to_string(writeLoopAccs) + "\n"
                  << std::endl;

        /*throw std::runtime_error(
            "incorrect number of read accesses to global buffer, expected: " + prod + " vs "
            + std::to_string(writeLoopAccs) + "\n");*/
        std::terminate();
    }
    if(static_cast<int>(numNodes.product()) != computeLoopAccs)
    {
        std::cout << " running into except 3" << std::endl;
        std::string prod = std::to_string(static_cast<int>(numNodes.product()));
        /*throw std::runtime_error(
            "incorrect number of computational Accesses to global buffer, expected: " + prod
            + " vs: " + std::to_string(computeLoopAccs) + "\n");*/
        std::terminate();
    }
    writeLoopAccs = 0;
    computeLoopAccs = 0;
}

struct StencilKernel2
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto const uCurrBuf,
        auto uNextBuf,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::CVector auto sharedMemExtents,
        alpaka::concepts::Vector auto numNodes,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using namespace alpaka;
        auto numFrames = acc[frame::count];
        auto frameExtent = acc[frame::extent];
        auto frameDomain = numFrames * frameExtent;
        auto traverseOverFrames = onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numFrames});
        using _2Vec = alpaka::Vec<u_int32_t, 2u>;
        auto const sMem = _2Vec{sharedMemExtents.x(), sharedMemExtents.y()};
        auto _0Vec = ALPAKA_TYPEOF(frameExtent){0u, 0u};
        auto const blockCount = acc[layer::thread].count();
        auto traverseOverExtentsWithHalo = onAcc::makeIdxMap(
            acc,
            onAcc::worker::threadsInBlock,
            IdxRange{
                Vec{0u, 0u},
                alpaka::Vec{std::min(frameExtent.x() + 2, sMem.x()), std::min(frameExtent.y() + 2, sMem.y())},
                Vec{1u, 1u}});
        auto traverseOverExtentsWithOutHalo = onAcc::makeIdxMap(
            acc,
            onAcc::worker::threadsInBlock,
            IdxRange{
                Vec{0u, 0u},
                alpaka::Vec{std::min(frameExtent.x(), sMem.x() - 2), std::min(frameExtent.y(), sMem.y() - 2)},
                Vec{1u, 1u}});


        for(auto frameIdx : traverseOverFrames)
        {
            for(auto bufStartIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::WorkerGroup{frameIdx, numFrames},
                    IdxRange{_0Vec, numNodes, frameExtent}))
            {
                onAcc::syncBlockThreads(acc);
                auto sdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

                for(auto elemIdxInFrame : traverseOverExtentsWithHalo)
                {
                    auto bufIdx = bufStartIdx + elemIdxInFrame;
                    sdata[elemIdxInFrame] = uCurrBuf[bufIdx];
                    writeLoopAccs++;
                }

                onAcc::syncBlockThreads(acc);
                double const rX = dt / (dx * dx);
                double const rY = dt / (dy * dy);

                constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
                constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};

                for(auto elemIdxInFrame : traverseOverExtentsWithOutHalo)
                {
                    auto idx2D = elemIdxInFrame + Vec{1u, 1u};
                    auto bufIdx = bufStartIdx + idx2D;
                    computeLoopAccs++;
                    uNextBuf[bufIdx] = sdata[idx2D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[idx2D - xDir] * rX
                                       + sdata[idx2D + xDir] * rX + sdata[idx2D - yDir] * rY
                                       + sdata[idx2D + yDir] * rY;
                }
            }
        }

        verifyCorrectness(numNodes, frameExtent);
    }
};

/*
for(alpaka::concepts::Dim<2u> auto blockStartIdx :
    onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, chunkSize}))
{
    auto sdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

    // avoid data race with the stencil calculation at the end
    onAcc::syncBlockThreads(acc);
    alpaka::Vec<std::size_t, 1u> h;
    for(alpaka::concepts::Dim<2u> auto idx2d :
        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
    {
        auto bufIdx = idx2d + blockStartIdx;
        sdata[idx2d] = uCurrBuf[bufIdx];
    }

    onAcc::syncBlockThreads(acc);

    // Each kernel executes one element
    double const rX = dt / (dx * dx);
    double const rY = dt / (dy * dy);

    constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
    constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};

    // go over only core cells
    // Vec{1, 1}; offset for halo above and to the left
    for(alpaka::concepts::Dim<2u> auto idx2D : onAcc::makeIdxMap(
            acc,
            onAcc::worker::threadsInBlock,
            IdxRange{chunkSize} >> 1u,
            onAcc::traverse::tiled))
    {
        auto bufIdx = idx2D + blockStartIdx;

        uNextBuf[bufIdx] = sdata[idx2D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[idx2D - xDir] * rX
                           + sdata[idx2D + xDir] * rX + sdata[idx2D - yDir] * rY + sdata[idx2D + yDir] * rY;
    }
}
*/
