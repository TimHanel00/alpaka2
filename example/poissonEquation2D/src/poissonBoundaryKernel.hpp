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

/*
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
template<typename Vec2>
struct PoissonBoundaryKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto p,
        auto extent,
        auto numNodes,
        double p0,
        double alpha,
        double dx) const -> void
    {
        using Idx = uint32_t;
        // Lower and upper edges: j = 0 and j = ny-1
        Idx extent_y = extent[0];
        Idx extent_x = extent[1];
        Idx nx = numNodes[1];
        for(Idx y = 0; y < extent_y; ++y)
        {
            p[Vec2{y, 0u}] = p0;
            p[Vec2{y, extent_x - 1u}] = p0 - alpha * (nx - 1.0) * dx;
        }
        for(Idx x = 1; x < extent_x - 1; ++x)
        {
            p[Vec2{0, x}] = p[Vec2{1, x}];
            p[Vec2{extent_y - 1u, x}] = p[Vec2{extent_y - 2u, x}];
        }
        /*
        // dirichlet boundary condition to make laminar flow work
        for(auto [y] : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec1{0u}, Vec1{extent_y}}))
        {
            p[Vec2{y, 0}] = p0;
            p[Vec2{y, extent_x - 1u}] = p0 - alpha * (nx - 1.0) * dx;
        }
        // Bottom and top edges Neumann boundary condition
        for(auto [x] :
            onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec1{0u}, Vec1{extent_x - 2u}} >> 1u))
        {
            p[Vec2{0, x}] = p[Vec2{1, x}];
            p[Vec2{extent_y - 1u, x}] = p[Vec2{extent_y - 2u, x}];
        }          */

// Left and right edges -- dirichlet boundary condition
// -- applying high pressure from the left and low pressure from the right (as a constant gradient) -- thus
// simulating laminar flow (as in the validation function)
// }
//};
template<typename Vec2>
struct PoissonBoundaryKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        auto p,
        auto extent,
        auto numNodes,
        double p0,
        double alpha,
        double dx) const -> void
    {
        using Idx = uint32_t;

        Idx extent_y = extent[0];
        Idx extent_x = extent[1];
        Idx nx = numNodes[1];
        Idx ny = numNodes[0];

        // Left and right boundaries (x = 0 and x = nx-1)
        for(Idx y = 0; y < extent_y; ++y)
        {
            p[Vec2{y, 0}] = p0;
            p[Vec2{y, extent_x - 1}] = p0 - alpha * (nx - 1.0) * dx;
        }

        // Top and bottom boundaries (y = 0 and y = ny-1)
        for(Idx x = 0; x < extent_x; ++x)
        {
            // Apply Dirichlet condition explicitly
            p[Vec2{0, x}] = p0;
            p[Vec2{extent_y - 1, x}] = p0;
        }
    }
};
#endif // POISSONBOUNDARYKERNEL_H
