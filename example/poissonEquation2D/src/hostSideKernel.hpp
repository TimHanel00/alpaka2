//
// Created by tim on 29.04.25.
//

#ifndef HOSTSIDEKERNEL_H
#define HOSTSIDEKERNEL_H
#include "alpaka/core/common.hpp"
#include "compareSerialImplementation.hpp"
#include "poissonBoundaryKernel.hpp"
#include "poissonStencil.hpp"
#include "residualKernel.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
using namespace alpaka;

template<
    typename T_FrameSpec,
    typename T_BorderFrameSpec,
    typename T_PressureBuf,
    typename T_NextPressureBuf,
    typename T_ResHost,
    typename T_ResAcc,
    typename T_bufHost,
    typename T_rhs,
    typename T_computeQueue,
    typename T_dumpQueue>
struct HostSideKernel
{
    T_FrameSpec& frameSpec;
    T_BorderFrameSpec& borderBlockingSpec;
    T_PressureBuf& pressureFieldBuffer;
    T_NextPressureBuf& nextPressureFieldBuffer;
    T_bufHost& bufHost;
    T_ResHost& residualHostBufElement;
    T_ResAcc& residualAccElement;
    T_rhs& rhs;

    T_computeQueue& computeQueue;
    T_dumpQueue& dumpQueue;
    HostSideKernel(
        T_FrameSpec& _spec,
        T_BorderFrameSpec& _borderBlockingSpec,
        T_PressureBuf& _bufAcc,
        T_NextPressureBuf& _nextBufAcc,
        T_bufHost& _bufHost,
        T_rhs& _rhs,
        T_ResHost& _resHost,
        T_ResAcc& _resAcc,
        T_computeQueue& _computeQueue,
        T_dumpQueue& _dumpQueue)
        : frameSpec(_spec)
        , borderBlockingSpec(_borderBlockingSpec)
        , pressureFieldBuffer(_bufAcc)
        , nextPressureFieldBuffer(_nextBufAcc)
        , bufHost(_bufHost)
        , rhs(_rhs)
        , residualHostBufElement(_resHost)
        , residualAccElement(_resAcc)
        , computeQueue(_computeQueue)
        , dumpQueue(_dumpQueue) {};

    template<typename TAcc>
    ALPAKA_FN_HOST auto operator()(
        TAcc const& acc,
        auto exec,
        alpaka::concepts::Vector auto const chunkSize,
        alpaka::concepts::CVector auto const sharedMemExtents,
        auto numNodes,
        auto halo,
        double dx,
        double dy,
        double const p0,
        double const alpha,
        auto omega,
        double tolerance,
        std::size_t cutoff) const -> void
    {
        using Vec2 = alpaka::Vec<uint32_t, 2u>;
        auto extent = numNodes + halo;

        std::cout << "omega: " << omega << " " << std::endl;
        // alpaka::onHost::wait(computeQueue);

        // applyBoundaryConditions<Vec2>(pressureFieldBuffer.getMdSpan(), extent, numNodes, p0, alpha, dx);
        alpaka::onHost::enqueue(
            computeQueue,
            exec,
            borderBlockingSpec,
            KernelBundle{
                PoissonBoundaryKernel<Vec2>{},
                pressureFieldBuffer.getMdSpan(),
                extent,
                numNodes,
                p0,
                alpha,
                dx});
        alpaka::onHost::wait(computeQueue);

        alpaka::onHost::enqueue(
            computeQueue,
            exec,
            frameSpec,
            alpaka::KernelBundle{
                computeResidualReduceKernel<Vec2>{},
                pressureFieldBuffer.getMdSpan(),
                rhs.getMdSpan(),
                residualAccElement.getMdSpan(),
                sharedMemExtents,
                chunkSize,
                numNodes,
                dx,
                dy});
        alpaka::onHost::memcpy(computeQueue, residualHostBufElement, residualAccElement);
        alpaka::onHost::wait(computeQueue);
        double initialNorm = sqrt(residualHostBufElement.getMdSpan()[0]);
        std::cout << "Initial norm: " << initialNorm << std::endl;

        std::size_t counter = 0;
        bool converged = false;
        double norm = 0.0;

        while(!converged && counter < cutoff)
        {
            /*
            applyStencilUpdate<Vec2>(
                pressureFieldBuffer.getMdSpan(),
                nextPressureFieldBuffer.getMdSpan(),
                rhs.getMdSpan(),
                extent,
                dx,
                dy,
                omega,
                pref);*/

            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                frameSpec,
                KernelBundle{
                    PoissonStencilKernel<Vec2>{},
                    pressureFieldBuffer.getMdSpan(),
                    nextPressureFieldBuffer.getMdSpan(),
                    rhs.getMdSpan(),
                    chunkSize,
                    sharedMemExtents,
                    numNodes,
                    dx,
                    dy,
                    omega[0]});

            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                borderBlockingSpec,
                KernelBundle{
                    PoissonBoundaryKernel<Vec2>{},
                    nextPressureFieldBuffer.getMdSpan(),
                    extent,
                    numNodes,
                    p0,
                    alpha,
                    dx});
            alpaka::onHost::wait(computeQueue);
            // applyBoundaryConditions<Vec2>(nextPressureFieldBuffer.getMdSpan(), extent, numNodes, p0, alpha, dx);
            std::swap(pressureFieldBuffer, nextPressureFieldBuffer);
            alpaka::onHost::memset(computeQueue, residualAccElement, 0x0);

            alpaka::onHost::enqueue(
                computeQueue,
                exec,
                frameSpec,
                alpaka::KernelBundle{
                    computeResidualReduceKernel<Vec2>{},
                    pressureFieldBuffer.getMdSpan(),
                    rhs.getMdSpan(),
                    residualAccElement.getMdSpan(),
                    sharedMemExtents,
                    chunkSize,
                    numNodes,
                    dx,
                    dy});

            alpaka::onHost::memcpy(computeQueue, residualHostBufElement, residualAccElement);
            alpaka::onHost::wait(computeQueue);
            double reducedResidual = residualHostBufElement.getMdSpan()[0];
            norm = sqrt(reducedResidual);
            // double norm = computeResidual(pressureFieldBuffer, rhs, extent, dx, dy);
            //  double norm = computeResidual(pressureFieldBuffer.getMdSpan(), rhs.getMdSpan(), extent, dx, dy);
            //   double norm = computeResidual(pressureFieldBuffer.getMdSpan(), rhs.getMdSpan(), extent, dx, dy);
            // std::cout << " current norm: " << norm << std::endl;
            if(norm < tolerance * initialNorm)
            {
                converged = true;
            }

            ++counter;
        }
        if(converged)
        {
            std::cout << " poisson equation converged after " << counter << " steps " << std::endl;
            std::cout << " Norm " << norm << " initialNorm " << initialNorm << std::endl;
        }
        if(counter >= cutoff)
        {
            std::cout << "unfortunately after " << counter << " runs no convergence in sight for Omega: " << omega[0]
                      << std::endl;
        }
    }
};
#endif // HOSTSIDEKERNEL_H
