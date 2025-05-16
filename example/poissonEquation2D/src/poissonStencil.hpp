/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "../example/heatEquation2D/src/helpers.hpp"

#include <alpaka/alpaka.hpp>

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
template<typename Vec2>
struct PoissonStencilKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const pressureField,
        alpaka::concepts::MdSpan auto nextPressureField,
        alpaka::concepts::MdSpan auto rhs,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::CVector auto sharedMemExtents,
        alpaka::concepts::Vector auto numNodes,
        double const dx,
        double const dy,
        double omega) const -> void
    {
        using namespace alpaka;


        for(alpaka::concepts::Dim<2u> auto blockStartIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, chunkSize}))
        {
            auto sdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            for(alpaka::concepts::Dim<2u> auto idx2d :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
            {
                auto bufIdx = idx2d + blockStartIdx;
                sdata[idx2d] = pressureField[bufIdx];
            }

            onAcc::syncBlockThreads(acc);
            auto factor = 0.9998;
            // Vec2 xDir{0u, 1u};
            // Vec2 yDir{1u, 0u};
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
                double lambda = 2.0;

                // Updated denominator of the Laplace operator (diagonal entry of A + λI)
                double denom = 2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy));

                // Updated prefactor based on new diagonal
                double pref = omega / denom;
                auto bufIdx = idx2D + blockStartIdx;
                double update = (sdata[idx2D + xDir] + sdata[idx2D - xDir]) / (dx * dx)
                                + (sdata[idx2D + yDir] + sdata[idx2D - yDir]) / (dy * dy) - rhs[bufIdx];
                // double diagCorrection = lambda * sdata[idx2D];
                nextPressureField[bufIdx] = (1 - omega) * factor * sdata[idx2D] + pref * update;
            }
        }
    }
};
