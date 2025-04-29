//
// Created by tim on 29.04.25.
//

#ifndef HOSTSIDEKERNEL_H
#define HOSTSIDEKERNEL_H
#include "alpaka/core/common.hpp"
#include "poissonBoundaryKernel.hpp"
#include "poissonStencil.hpp"
#include "residualKernel.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
using namespace alpaka;

struct HostSideKernel
{
    template<typename TAcc>
    ALPAKA_FN_HOST auto operator()(
        TAcc const& acc,
        auto devAcc,
        auto& computeQueue,
        auto exec,
        auto frameSpec,
        auto borderKernelSpec,
        auto pressureFieldBuffer,
        auto nextPressureFieldBuffer,
        alpaka::concepts::MdSpan auto rhs,
        alpaka::concepts::MdSpan auto residual,
        auto residualHostBuf,
        auto residualAccBuf,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::CVector auto const sharedMemExtents,
        alpaka::concepts::Vector auto const numNodesWithHalo,
        alpaka::concepts::Vector auto numNodes,
        double const dx,
        double const dy,
        double_t omega,
        double tolerance) const -> void
    {
        PoissonStencilKernel stencil_kernel;
        PoissonBoundaryKernel boundary_kernel;
        computeResidualReduceKernel residualReduceKernel;
        double pref = omega / (2 * (1. / (dx * dx) + 1. / (dy * dy))); // constant prefactor in the sor formula
        double n; // norm of initial residuum
        //    double r[nx*ny]; // residuum
        bool converged = false; // boolean indicating the abortion criterion
        int counter = 0; // counter to abort if it gets stuck
        int cutoff = 1000; // maximum step number before abortion

        onHost::Queue dumpQueue = devAcc.makeQueue();
        alpaka::onHost::enqueue(
            computeQueue,
            exec,
            frameSpec,
            alpaka::KernelBundle{
                residualReduceKernel,
                residual,
                pressureFieldBuffer.getMdSpan(),
                rhs,
                sharedMemExtents,
                chunkSize,
                numNodes,
                residualAccBuf.getMdSpan(),
                dx,
                dy});
        alpaka::onHost::memcpy(dumpQueue, residualHostBuf, residualAccBuf);
        alpaka::onHost::wait(dumpQueue);
        double reducedResidual = residualHostBuf.getMdSpan()[0];
        double initialNorm = std::sqrt(reducedResidual);
        while(!converged && counter < cutoff)
        {
            // Compute next values
            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                frameSpec,
                alpaka::KernelBundle{
                    stencil_kernel,
                    pressureFieldBuffer.getMdSpan(),
                    nextPressureFieldBuffer.getMdSpan(),
                    rhs,
                    chunkSize,
                    sharedMemExtents,
                    numNodes,
                    dx,
                    dy,
                    omega,
                    pref});

            // Apply boundaries
            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                borderKernelSpec,
                alpaka::KernelBundle{boundary_kernel, nextPressureFieldBuffer.getMdSpan(), numNodesWithHalo});

            // So we just swap next and curr (shallow copy)
            std::swap(pressureFieldBuffer, nextPressureFieldBuffer);
            // recalculate residualNorm
            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                frameSpec,
                alpaka::KernelBundle{
                    residualReduceKernel,
                    residual,
                    pressureFieldBuffer.getMdSpan(),
                    rhs,
                    sharedMemExtents,
                    chunkSize,
                    numNodes,
                    residualAccBuf.getMdSpan(),
                    dx,
                    dy});
            alpaka::onHost::memcpy(dumpQueue, residualHostBuf, residualAccBuf);
            alpaka::onHost::wait(dumpQueue);
            reducedResidual = residualHostBuf.getMdSpan()[0];
            double norm = std::sqrt(reducedResidual);
            if(norm < tolerance * initialNorm)
            {
                converged = true;
            }
            ++counter;
        }
    }
};
#endif // HOSTSIDEKERNEL_H
