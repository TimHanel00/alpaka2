//
// Created by tim on 29.04.25.
//

#ifndef POISSONBOUNDARYKERNEL_H
#define POISSONBOUNDARYKERNEL_H
/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Applies boundary conditions
//! forward difference in t and second-order central difference in x
//!
//! \param uBuf grid values of u for each x, y and the current value of t:
//!                 u(x, y, t)  | t = t_current
//! \param chunkSize
//! \param pitch
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct PoissonBoundaryKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,

        alpaka::concepts::MdSpan auto uBuf,
        alpaka::concepts::Vector auto numNodesWithHalo) const -> void
    {
        using Idx = uint32_t;

        using namespace alpaka;

        using Idx = uint32_t;
        using Vec = alpaka::Vec<Idx, 2>;
        constexpr auto xDir = CVec<uint32_t, 0u, 1u>{};
        constexpr auto yDir = CVec<uint32_t, 1u, 0u>{};
        // Lower and upper edges: j = 0 and j = ny-1
        Idx const nx = numNodesWithHalo[0];
        Idx const ny = numNodesWithHalo[1];

        // Bottom and top edges
        for(auto idx :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{0u, 0u}, Vec{0u, nx - 2u}} >> 1u))
        {
            Vec bottom = idx - yDir;
            Vec top = idx + Vec{ny - 2u, 0u};
            Vec topEdge = top + yDir;

            uBuf[bottom] = uBuf[idx];
            uBuf[topEdge] = uBuf[top];
        }

        // Left and right edges
        for(auto idx :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{0u, 0u}, Vec{nx - 2u, 0u}} >> 1u))
        {
            Vec leftEdge = idx - xDir;
            Vec right = Vec{0u, nx - 2u} + idx;
            Vec rightEdge = right + xDir;

            uBuf[leftEdge] = uBuf[idx];
            uBuf[rightEdge] = uBuf[right];
        }
    }
};

#endif // POISSONBOUNDARYKERNEL_H
