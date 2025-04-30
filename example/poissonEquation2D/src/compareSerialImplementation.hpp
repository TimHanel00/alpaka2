#ifndef COMPARESERIALIMPLEMENTATION_H
#define COMPARESERIALIMPLEMENTATION_H

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <iostream>

template<typename Vec2>
inline std::size_t idx2d(std::size_t y, std::size_t x, std::size_t strideX)
{
    return y * strideX + x;
}

template<typename Vec2>
void applyBoundaryConditions(auto p, auto extent, auto numNodes, double p0, double alpha, double dx)
{
    using T_idx = uint32_t;
    T_idx extent_y = extent[0];
    T_idx extent_x = extent[1];
    T_idx nx = numNodes[1];

    for(T_idx y = 0; y < extent_y; ++y)
    {
        p[Vec2{y, 0u}] = p0;
        p[Vec2{y, extent_x - 1u}] = p0 - alpha * (nx - 1.0) * dx;
    }
    for(T_idx x = 1; x < extent_x - 1; ++x)
    {
        p[Vec2{0, x}] = p[Vec2{1, x}];
        p[Vec2{extent_y - 1u, x}] = p[Vec2{extent_y - 2u, x}];
    }
}

double computeResidual(auto p, auto rhs, auto const extent, double const dx, double const dy)
{
    using TIdx = uint32_t;
    using Vec2 = alpaka::Vec<TIdx, 2u>;
    // static_assert(std::is_same_v<decltype(extent[0]), void>);
    TIdx extent_y = extent[0];
    TIdx extent_x = extent[1];
    double accum = 0.0;
    Vec2 const xDir{0u, 1u};
    Vec2 const yDir{1u, 0u};
    for(auto y = TIdx{1}; y < extent_y - TIdx{1}; ++y)
    {
        for(auto x = TIdx{1}; x < extent_x - TIdx{1}; ++x)
        {
            Vec2 idx{y, x};
            double laplacian = (p[idx + xDir] - 2.0 * p[idx] + p[idx - xDir]) / (dx * dx)
                               + (p[idx + yDir] - 2.0 * p[idx] + p[idx - yDir]) / (dy * dy);
            double h = laplacian - rhs[idx];
            accum += (h * h);
        }
    }
    return std::sqrt(accum);
}

template<typename Vec2>
void applyStencilUpdate(
    auto pressure,
    auto next_pressure,
    auto rhs,
    Vec2 const& extent,
    double dx,
    double dy,
    double omega,
    double pref)
{
    using TIdx = uint32_t;
    TIdx extent_y = extent[0];
    TIdx extent_x = extent[1];

    for(TIdx y = 1u; y < extent_y - 1u; ++y)
    {
        for(TIdx x = 1u; x < extent_x - 1u; ++x)
        {
            Vec2 idx{y, x};
            double update = (pressure[Vec2{y, x + 1u}] + pressure[Vec2{y, x - 1u}]) / (dx * dx)
                            + (pressure[Vec2{y + 1u, x}] + pressure[Vec2{y - 1u, x}]) / (dy * dy) - rhs[idx];
            next_pressure[idx] = (1.0 - omega) * pressure[idx] + pref * update;
        }
    }
}

void solvePoissonSerialGeneric(
    auto pressure,
    auto next_pressure,
    auto rhs,
    auto numNodes,
    auto halo,
    double dx,
    double dy,
    double p0,
    double alpha,
    double omega,
    double tolerance = 1e-4,
    std::size_t cutoff = 1000)
{
    using Vec2 = alpaka::Vec<uint32_t, 2>;
    auto extent = numNodes + halo;
    double pref = omega / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)));

    applyBoundaryConditions<Vec2>(pressure, extent, numNodes, p0, alpha, dx);

    double initialNorm = computeResidual<Vec2>(pressure, rhs, extent, dx, dy);
    std::cout << "Initial norm: " << initialNorm << std::endl;

    std::size_t counter = 0;
    bool converged = false;

    while(!converged && counter < cutoff)
    {
        applyStencilUpdate<Vec2>(pressure, next_pressure, rhs, extent, dx, dy, omega, pref);
        applyBoundaryConditions<Vec2>(next_pressure, extent, numNodes, p0, alpha, dx);
        std::swap(pressure, next_pressure);

        double norm = computeResidual<Vec2>(pressure, rhs, extent, dx, dy);
        std::cout << " Norm " << norm << " initialNorm " << initialNorm << std::endl;

        if(norm < tolerance * initialNorm)
        {
            std::cout << " poisson equation converged after " << counter << " steps " << std::endl;
            std::cout << " Norm " << norm << " initialNorm " << initialNorm << std::endl;
            converged = true;
        }

        ++counter;
    }

    if(counter >= cutoff)
    {
        std::cout << "unfortunately after " << counter << " runs no convergence in sight for Omega: " << omega
                  << std::endl;
    }
}

#endif // COMPARESERIALIMPLEMENTATION_H
