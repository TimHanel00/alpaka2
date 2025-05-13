/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */
#include "BoundaryKernel2.hpp"
#include "analyticalSolution2.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#ifdef PNGWRITER_ENABLED
#    include "writeImage.hpp"
#endif

#include "userDefTraitsForDynSmem.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>

/*
template<typename T_Kernel, typename T_Spec>
struct BlockDynSharedMemBytes;

template<typename T_numFrames, typename T_numThreads>
struct BlockDynSharedMemBytes<StencilKernel2, alpaka::onHost::FrameSpec<T_numFrames, T_numThreads>>
{
    BlockDynSharedMemBytes(StencilKernel2 const&, alpaka::onHost::FrameSpec<T_numFrames, T_numThreads> spec)
        : spec_(spec)
    {
    }

    template<typename TExec, typename... Floats>
    uint32_t operator()(TExec const&, Floats const&...) const
    {
        std::cout << " executed Kernel with sMEM " << spec_.m_frameExtent.x() << " " << spec_.m_frameExtent.y()
                  << std::endl;
        return static_cast<uint32_t>(spec_.m_frameExtent.x() * spec_.m_frameExtent.y() * sizeof(double)+2);
    }

    alpaka::onHost::FrameSpec<T_numFrames, T_numThreads> spec_;
};
*/
// namespace alpaka::onHost::trait

//! Each kernel computes the next step for one point.
//! Therefore the number of threads should be equal to numNodesX.
//! Every time step the kernel will be executed numNodesX-times
//! After every step the curr-buffer will be set to the calculated values
//! from the next-buffer.
//!
//! In standard projects, you typically do not execute the code with any available accelerator.
//! Instead, a single accelerator is selected once from the active accelerators and the kernels are executed
//! with the selected accelerator only. If you use the example as the starting point for your project, you can
//! rename the example() function to main() and move the accelerator tag to the function body.
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

    // simulation defines
    // {Y, X}

    // withGridSizeTune(tune::GridSizeTune{fVec{108}, IdxRange{fVec{108}, dataBlocking.m_numFrames,
    // fVec{108}}}). withGridSizeTune(tune::GridSizeTune{fVec{22}, IdxRange{fVec{22}, fVec{32},
    // fVec{1}}}).//#cpu

    constexpr IdxVec numNodes{16 * 1024, 16 * 1024};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000 * 32 * 4;
    constexpr double tMax = 0.00000005;
    // GPU settings
    /*
    constexpr IdxVec numNodes{128, 128};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000 * 2;
    constexpr double tMax = 0.001;
    */
    // x, y in [0, 1], t in [0, tMax]
    constexpr double dx = 1.0 / static_cast<double>(extent[1] - 1);
    constexpr double dy = 1.0 / static_cast<double>(extent[0] - 1);
    constexpr double dt = tMax / static_cast<double>(numTimeSteps);

    // Check the stability condition
    double r = 2 * dt / ((dx * dx * dy * dy) / (dx * dx + dy * dy));
    if(r > 1.)
    {
        std::cerr << "Stability condition check failed: dt/min(dx^2,dy^2) = " << r
                  << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

    // Initialize host-buffer
    // This buffer will hold the current values (used for the next step)
    auto uBufHost = alpaka::onHost::alloc<double>(devHost, extent);

    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);
    auto uNextBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);

    auto const pitchCurrAcc{uCurrBufAcc.getPitches()};
    auto const pitchNextAcc{uNextBufAcc.getPitches()};

    // Set buffer to initial conditions
    initalizeBuffer(uBufHost.getMdSpan(), dx, dy);

    // Select queue
    Queue dumpQueue = devAcc.makeQueue();
    Queue computeQueue = devAcc.makeQueue();

    // Copy host -> device
    alpaka::onHost::memcpy(computeQueue, uCurrBufAcc, uBufHost);
    alpaka::onHost::wait(computeQueue);

    // Appropriate chunk size to split your problem for your Acc
    constexpr Idx xSize = 32u;
    constexpr Idx ySize = 32u;
    constexpr Idx halo = 2u;
    constexpr auto chunkSize = CVec<Idx, ySize, xSize>{};
    constexpr auto numNodesWithHalo = numNodes + halo;

    constexpr IdxVec numChunks{
        divCeil(numNodes, IdxVec{xSize, ySize}),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");

    auto sharedMemExtents = CVec<uint32_t, ySize + halo, xSize + halo>{};
    StencilKernel2 stencilKernel;
    BoundaryKernel2 boundaryKernel;

    auto dataBlockingStencil = FrameSpec{numChunks, chunkSize};
    constexpr auto longestSide = std::max(numNodesWithHalo.y(), numNodesWithHalo.x());
    auto dataBlockingBorder = FrameSpec{Vec{longestSide / chunkSize.x()}, Vec{std::max(chunkSize.y(), chunkSize.x())}};
    auto toRTime = FrameSpec{
        alpaka::Vec{dataBlockingStencil.m_numFrames.x(), dataBlockingStencil.m_numFrames.y()},
        Vec{dataBlockingStencil.m_frameExtent.x(), dataBlockingStencil.m_frameExtent.y()}};
    using fVec = ALPAKA_TYPEOF(toRTime.m_frameExtent);
    using uVec = ALPAKA_TYPEOF(toRTime.m_numFrames);
    //  builder.build();
    /*
    auto tuningSession
        = builder.withStrategy(alpaka::tune::strategy::randomSearch{})
              .withBlockSizeTune(
                  tune::ThreadBlockSizeTune{
                      toRTime.m_frameExtent,
                      IdxRange{fVec{8, 4}, toRTime.m_frameExtent, fVec{8, 4}}})
              .withNumBlocksTune(
                  tune::NumBlocksTune{fVec{7, 8}, IdxRange{fVec{7, 8}, toRTime.m_numFrames, fVec{7, 8}}})
              .withConfig("./config/babelstream.toml")
              .build(); // #gpu*/
    /*
    auto tuningSession
        = builder.withStrategy(alpaka::tune::strategy::randomSearch{})
              .withBlockSizeTune(
                  alpaka::tune::ThreadBlockSizeTune{
                      fVec{32, 32},
                      IdxRange{fVec{32, 32}, toRTime.m_frameExtent, fVec{32, 32}}})
              .withNumBlocksTune(
                  alpaka::tune::NumBlocksTune{uVec{56, 56}, IdxRange{uVec{56, 56}, toRTime.m_numFrames,
    uVec{56, 56}}}) .withConfig("./config/babelstream.toml") .build(); // #gpu*/

    using VecType = ALPAKA_TYPEOF(dataBlockingBorder.m_numFrames);

    auto setFixedNumBlocks = uVec{3, 4};
    auto setFixedNumThreads = fVec{16, 16};

    auto setFixedNumBlocks_
        = tune::primeFactorPartitioning(onHost::getDeviceProperties(devAcc).m_multiProcessorCount * 16u, uVec{});
    auto setnumThreads_ = fVec{4, 8};
    auto frameSpec = FrameSpec{toRTime.m_numFrames, toRTime.m_frameExtent, toRTime.m_numFrames, toRTime.m_frameExtent};
    std::cout << " original numFrames " << frameSpec.m_numFrames.toString() << std::endl;
    auto tuningSession
        = tune::TuningBuilder{} //.withBlockSizeTune()

              //.withNumFramesTune(
              // tune::Tuneable(uVec{64, 64}, setFixedNumBlocks_, toRTime.m_numFrames, setFixedNumBlocks_))
              //.withFrameExtentTune(tune::Tuneable(fVec{8, 4}, toRTime.m_frameExtent, fVec{8, 4}))
              //.withNumBlocksTune(setFixedNumBlocks_,IdxRange{fVec{4, 8}, toRTime.m_frameExtent * fVec{4, 4}, fVec{4,
              // 8}})
              /*
                        .withFrameExtentTune(
                            tune::Tuneable(
                                setnumThreads_,
                                IdxRange{setnumThreads_, toRTime.m_frameExtent * fVec{4, 4}, fVec{2, 2}}))
                  */
              .withFrameExtentTune(tune::Tuneable(IdxRange{fVec{4, 8}, toRTime.m_frameExtent, fVec{4, 8}}))
              .withNumBlocksTune(
                  tune::Tuneable(IdxRange{setFixedNumBlocks_, toRTime.m_numFrames * uVec{2, 2}, setFixedNumBlocks_}))
              .withBlockSizeTune(tune::Tuneable(IdxRange{fVec{4, 8}, toRTime.m_frameExtent, fVec{4, 8}}))
              .template withConstraint<tune::frameTune::numBlocks, tune::frameTune::FrameExtent>(
                  [numNodes](auto numBlocks, auto numElementsPerChunk)
                  {
                      bool XinBounds = numBlocks.x() * numElementsPerChunk.x() <= numNodes.x();
                      bool YinBounds = numBlocks.y() * numElementsPerChunk.y() <= numNodes.y();
                      return XinBounds && YinBounds;
                  })

              .template withConstraint<tune::frameTune::FrameExtent>(
                  [numNodes](auto a)
                  {
                      using type = std::remove_cvref_t<decltype(a.x())>;
                      auto condX = (numNodes.x() % a.x()) == type{0};
                      auto condY = (numNodes.y() % a.y()) == type{0};

                      return condX && condY;
                  })
              /*
              .template withConstraint<tune::frameTune::FrameExtent, tune::frameTune::ThreadBlock>(
                  [numNodes, setFixedNumBlocks_](auto frameExtent, auto threadBlockExtent)
                  {
                      using type = std::remove_cvref_t<decltype(frameExtent.x())>;
                      auto condX = (numNodes.x() % frameExtent.x()) == type{0};
                      auto condY = (numNodes.y() % frameExtent.y()) == type{0};
                      auto condZ
                          = frameExtent.x() >= threadBlockExtent.x() && frameExtent.y() >= threadBlockExtent.y();
                      auto condU = ((setFixedNumBlocks_.x() * frameExtent.x()) <= numNodes.x())
                                   && (setFixedNumBlocks_.y() * frameExtent.y()) <= numNodes.y();
                      return condX && condY && condZ && condU;
                  }) /*
               .template withConstraint<tune::frameTune::FrameExtent, tune::frameTune::ThreadBlock>(
                   [](auto a, auto b)
                   {
                       using type = std::remove_cvref_t<decltype(a.x())>;
                       auto condZ = (a.x() >= b.x() & a.y() >= b.y());

                       return condZ;
                   })*/
              .withStrategy(alpaka::tune::strategy::exhaustiveSearch{})
              /*
              .template withConstraint<tune::frameTune::NumFrames, tune::frameTune::FrameExtent>(
              [toRTime, numNodes](auto a, auto b) { return numNodes > (a * b); })*/


              .withConfig("./config/babelstream.toml")
              .build();
    auto startTime = std::chrono::high_resolution_clock::now();

    // Simulate
    for(uint32_t step = 1; step <= numTimeSteps; ++step)
    {
        // Compute next values
        /*
        alpaka::onHost::enqueue(
            computeQueue,
            exec,
            toRTime,
            KernelBundle{
                stencilKernel,
                uCurrBufAcc.getMdSpan(),
                uNextBufAcc.getMdSpan(),
                chunkSize,
                sharedMemExtents,
                numNodes,
                dx,
                dy,
        */
        tuningSession.enqueue(
            devAcc,
            computeQueue,
            exec,
            frameSpec,
            KernelBundle{stencilKernel, uCurrBufAcc.getMdSpan(), uNextBufAcc.getMdSpan(), numNodes, dx, dy, dt});
        // Apply boundaries
        alpaka::onHost::enqueue(
            computeQueue,
            exec,
            dataBlockingBorder,
            KernelBundle{boundaryKernel, uNextBufAcc.getMdSpan(), chunkSize, numNodesWithHalo, step, dx, dy, dt});

#ifdef PNGWRITER_ENABLED
        if((step - 1) % 100 == 0)
        {
            alpaka::onHost::wait(computeQueue);
            alpaka::onHost::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
            alpaka::onHost::wait(dumpQueue);
            writeImage(step - 1, uBufHost.getMdSpan());
        }
#endif

        // So we just swap next and curr (shallow copy)
        std::swap(uNextBufAcc, uCurrBufAcc);
    }

    alpaka::onHost::wait(computeQueue);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "Simulation took " << elapsedTime.count() << " seconds." << std::endl;


    // Copy device -> host
    alpaka::onHost::memcpy(dumpQueue, uBufHost, uCurrBufAcc);
    alpaka::onHost::wait(dumpQueue);

    // Validate
    auto const [resultIsCorrect, maxError] = validateSolution(uBufHost.getMdSpan(), extent, dx, dy, tMax);

    if(resultIsCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Max error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
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
};
