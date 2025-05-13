/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#include "analyticalSolution_poisson.hpp"
#include "hostSideKernel.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#ifdef PNGWRITER_ENABLED
#    include "../example/heatEquation2D/src/writeImage.hpp"
#endif

#include "compareSerialImplementation.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

template<size_t N>
constexpr char getFirstChar(char const (&str)[N])
{
    return str[0]; // ✅ OK, compile-time access
}

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
//! selected accelerator only. If you use the example as the starting point for your project, you can rename the
//! example() function to main() and move the accelerator tag to the function body.
template<typename T_Cfg>
auto example(T_Cfg const& cfg) -> int
{
    using namespace alpaka;
    using namespace alpaka::onHost;

    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for " << api.getName() << std::endl;

    // Select specific devices
    Platform platformHost = makePlatform(api::cpu);
    Device devHost = platformHost.makeDevice(0);

    Platform platformAcc = makePlatform(api);
    Device devAcc = platformAcc.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
    // support for double precision is not guaranteed for sycl devices such as Intel GPUs
    if constexpr(std::is_same_v<decltype(api), api::SyclIntelGpu>)
    {
        if(devAcc.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
        {
            std::cout << onHost::getName(devAcc) << " does not support double precision"
                      << "\n";
            std::cout << "Skip benchmark.\n";
            std::cout << "For Intel Arc GPUs, use the environemnt variables `IGC_EnableDPEmulation=1 "
                         "OverrideDefaultFP64Settings=1` to emulate double precision support.\n";
            // return 0 otherwise ctest fails
            return EXIT_SUCCESS;
        }
    }
#endif

    // simulation defines
    // {Y, X}
    constexpr IdxVec numNodes{256, 256};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000 * 32;
    constexpr double tMax = 0.000001;

    std::vector<double> convChange={1.0,10.0,500.0,10000.0};
    for(int j=0;j<convChange.size();++j){
        // x, y in [0, 1], t in [0, tMax]
        double dx = convChange[j] / static_cast<double>(extent[1] - 1);
        double dy = convChange[j] / static_cast<double>(extent[0] - 1);
        // Initialize host-buffer
        // This buffer will hold the current values (used for the next step)
        auto uBufHost = alpaka::onHost::alloc<double>(devHost, extent);
        auto reducedResidual_HostBuf = alpaka::onHost::alloc<double>(devHost, alpaka::Vec<std::size_t, 1>{1});
        reducedResidual_HostBuf[0] = 0.0;
        // Accelerator buffers
        auto uCurrBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);
        auto uNextBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);
        auto residual = alpaka::onHost::allocMirror(devAcc, uBufHost);
        auto rhs = alpaka::onHost::allocMirror(devAcc, uBufHost); // right hand side of the equation --> f(x,y)
        auto reducedResidual_AccBuf = alpaka::onHost::allocMirror(devAcc, reducedResidual_HostBuf);
        // Set buffer to initial conditions
        initalizeBuffer(uBufHost.getMdSpan(), dx, dy);
        Queue computeQueue = devAcc.makeQueue();
        Queue dumpQueue = devAcc.makeQueue();
        // Copy host -> device
        alpaka::onHost::memcpy(computeQueue, rhs, uBufHost);
        alpaka::onHost::memcpy(computeQueue, reducedResidual_AccBuf, reducedResidual_HostBuf);
        alpaka::onHost::memcpy(computeQueue, uCurrBufAcc, uBufHost);
        alpaka::onHost::memcpy(
            computeQueue,
            uNextBufAcc,
            uBufHost); // for corner cells since they are not handled in the boundary Kernel
        alpaka::onHost::wait(computeQueue);

        // Appropriate chunk size to split your problem for your Acc
        constexpr Idx xSize = 16u;
        constexpr Idx ySize = 16u;
        constexpr Idx halo = 2u;
        constexpr auto chunkSize = CVec<Idx, ySize, xSize>{};
        constexpr auto numNodesWithHalo = numNodes + halo;

        IdxVec numChunks{
            alpaka::divCeil(numNodes[0], chunkSize[0]),
            alpaka::divCeil(numNodes[1], chunkSize[1]),
        };

        assert(
            numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
            && "Domain must be divisible by chunk size");

        auto sharedMemExtents = CVec<uint32_t, ySize + halo, xSize + halo>{};
        // acceptLiteral("daw");
        auto dataBlockingStencil = FrameSpec{numChunks, chunkSize};
        constexpr auto longestSide = std::max(numNodesWithHalo.y(), numNodesWithHalo.x());
        auto dataBlockingBorder = FrameSpec{Vec{longestSide / chunkSize.x()}, Vec{std::max(chunkSize.y(), chunkSize.x())}};
        auto toRTime = FrameSpec{
            Vec{dataBlockingStencil.m_numFrames.x(), dataBlockingStencil.m_numFrames.y()},
            Vec{dataBlockingStencil.m_frameExtent.x(), dataBlockingStencil.m_frameExtent.y()}};
        using uVec = ALPAKA_TYPEOF(toRTime.m_numFrames);
        using fVec = ALPAKA_TYPEOF(toRTime.m_frameExtent);
        // static_assert(std::is_same_v<uVec, void>);
        // static_assert(std::is_same_v<fVec, void>);
        auto vec = fVec{4, 8}; // does not work.

        auto tuningSession = tune::TuningBuilder{}
                                 .withStrategy(alpaka::tune::strategy::iterativeRefinement{})
                                 .withRunSpecifiers(std::to_string(convChange[j]))
                                 .withConfig("./config/poissonEqu"+std::to_string(convChange[j])+".toml")
                                 .build();
        std::cout << " after build " << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        Queue hostQueue = devHost.makeQueue();
        HostSideKernel host_side_kernel{
            toRTime,
            dataBlockingBorder,
            uCurrBufAcc,
            uNextBufAcc,
            uBufHost,
            rhs,
            reducedResidual_HostBuf,
            reducedResidual_AccBuf,
            computeQueue,
            dumpQueue};
        using Vec1_float = alpaka::Vec<std::double_t, 1>;
        constexpr double tolerance = 1e-3;
        constexpr double p0 = 1.0;
        constexpr double alpha = 1.0;

        constexpr std::size_t cutoff = 10000;
        constexpr double omega = 1.95;
        constexpr double numRuns = 200;
        // solvePoissonSerialGeneric(uBufHost, uNextBufAcc, rhs, numNodes, halo, dx, dy, p0, alpha, omega); // serial
        //  IMplementation
        for(int i = 0; i < numRuns; i++)
        {
            alpaka::onHost::memset(computeQueue, uCurrBufAcc, 0x0);
            alpaka::onHost::memset(computeQueue, uNextBufAcc, 0x0);
            alpaka::onHost::wait(computeQueue);
            tuningSession.enqueue(
                devHost,
                hostQueue,
                alpaka::exec::CpuSerial{},
                FrameSpec{alpaka::Vec{1}, alpaka::Vec{1}},
                // KernelBundle{host_side_kernel, exec});
                KernelBundle{
                    host_side_kernel,
                    exec,
                    chunkSize,
                    sharedMemExtents,
                    numNodes,
                    halo,
                    dx,
                    dy,
                    p0,
                    alpha,
                    // omega,
                    alpaka::tune::Tuneable{
                        Vec1_float{1.95},
                        IdxRange{Vec1_float{1.95}, Vec1_float{2.0}, Vec1_float{0.005}},
                        "Omega"},
                    tolerance,
                    cutoff});
        }


    // omega});

    alpaka::onHost::wait(hostQueue);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Simulation took " << elapsedTime.count() << " seconds." << std::endl;


    // Copy device -> host
    alpaka::onHost::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::onHost::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost.getMdSpan(), extent, dx, dy, p0, alpha);
        if(resultIsCorrect)
        {
            std::cout << "Execution results correct!" << std::endl;
            //return EXIT_SUCCESS;
        }
        else
        {
            std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                      << std::endl;
            //return EXIT_FAILURE;
        }
    }


}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}
