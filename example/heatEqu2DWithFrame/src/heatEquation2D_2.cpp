/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber,
 * Tapish Narwal
 * SPDX-License-Identifier: ISC
 */
#include "BoundaryKernel2.hpp"
#include "analyticalSolution2.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <alpaka/onHost/internal.hpp>
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
template<typename Exec_T>
static auto getSessionFromExec(Exec_T const& exec, auto frameSpec, auto& devAcc, auto numNodes)
{
    using namespace alpaka;
    using fVec = ALPAKA_TYPEOF(frameSpec.m_frameExtent);
    using uVec = ALPAKA_TYPEOF(frameSpec.m_numFrames);
    auto tuningSession
        = tune::TuningBuilder{}
              .withFrameExtentTune(tune::Tuneable(IdxRange{fVec{4, 8}, frameSpec.m_frameExtent, fVec{4, 8}}))
              .withNumBlocksTune()
              .withBlockSizeTune(tune::Tuneable(IdxRange{fVec{4, 8}, frameSpec.m_frameExtent, fVec{4, 8}}))
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
              .withStrategy(tune::strategy::exhaustiveSearch{})
              /*
              .template withConstraint<tune::frameTune::NumFrames, tune::frameTune::FrameExtent>(
              [toRTime, numNodes](auto a, auto b) { return numNodes > (a * b); })*/
              .withConfig("./config/heatEquation_GPU_" + numNodes.toString() + ".toml")
              .build();
    return tuningSession;
};

static auto getSessionFromExec(alpaka::exec::CpuOmpBlocks const& exec, auto frameSpec, auto& devAcc, auto numNodes)
{
    using namespace alpaka;
    using fVec = ALPAKA_TYPEOF(frameSpec.m_frameExtent);
    using uVec = ALPAKA_TYPEOF(frameSpec.m_numFrames);
    auto tuningSession
        = tune::TuningBuilder{}
              .withFrameExtentTune(tune::Tuneable(IdxRange{fVec{4, 4}, frameSpec.m_frameExtent, fVec{4, 4}}))
              .withNumBlocksTune()
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
              .withStrategy(tune::strategy::exhaustiveSearch{})
              /*
              .template withConstraint<tune::frameTune::NumFrames, tune::frameTune::FrameExtent>(
              [toRTime, numNodes](auto a, auto b) { return numNodes > (a * b); })*/


              .withConfig("./config/heatEquation_CPU" + numNodes.toString() + ".toml")
              .build();
    return tuningSession;
};

template<typename T_Exec>
constexpr auto getNumNodes(T_Exec const& exec)
{
    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;
    constexpr IdxVec numNodes{16 * 1024, 16 * 1024};
    return numNodes;
}

constexpr auto getNumNodes(alpaka::exec::CpuOmpBlocks const& exec)
{
    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;
    constexpr IdxVec numNodes{16 * 1024, 16 * 1024};
    return numNodes;
}

template<typename T_TuningSession>
bool abortIfFinished(T_TuningSession const& session)
{
    if(session.finishedConfigs >= 1)
    {
        return true;
    };
    return false;
}

void log_event()
{
    static std::string label = "Stencil";
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() % 1'000'000'000;

    std::tm* tm = std::localtime(&t_c);
    std::cout << "[" << std::put_time(tm, "%F %T") << "." << std::setfill('0') << std::setw(9) << ns << "]," << label
              << std::endl;
}

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
auto example(T_Cfg const& cfg, uint32_t i) -> int
{
    using namespace alpaka;
    using namespace alpaka::onHost;

    using Idx = uint32_t;
    using IdxVec = alpaka::Vec<Idx, 2u>;
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);
    // simulation defines
    // {Y, X}

    // withGridSizeTune(tune::GridSizeTune{fVec{108}, IdxRange{fVec{108}, dataBlocking.m_numFrames,
    // fVec{108}}}). withGridSizeTune(tune::GridSizeTune{fVec{22}, IdxRange{fVec{22}, fVec{32},
    // fVec{1}}}).//#cpu

    IdxVec numNodes = {i, i};
    accessArraySize<IdxVec>(numNodes);
    constexpr IdxVec haloSize{2, 2};
    IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000 * 32 * 4;
    constexpr double tMax = 0.000005;
    // GPU settings
    /*
    constexpr IdxVec numNodes{128, 128};
    constexpr IdxVec haloSize{2, 2};
    constexpr IdxVec extent = numNodes + haloSize;

    constexpr uint32_t numTimeSteps = 4000 * 2;
    constexpr double tMax = 0.001;
    */
    // x, y in [0, 1], t in [0, tMax]
    double dx = 1.0 / static_cast<double>(extent[1] - 1);
    double dy = 1.0 / static_cast<double>(extent[0] - 1);
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
    auto uBufHost = alpaka::onHost::allocHost<double>(extent);

    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);
    auto uNextBufAcc = alpaka::onHost::allocMirror(devAcc, uBufHost);
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
    auto numNodesWithHalo = numNodes + halo;


    IdxVec numChunks{
        divCeil(numNodes, IdxVec{xSize, ySize}),
    };

    assert(
        numNodes[0] % chunkSize[0] == 0 && numNodes[1] % chunkSize[1] == 0
        && "Domain must be divisible by chunk size");
    StencilKernel2 stencilKernel;
    BoundaryKernel2 boundaryKernel;
    auto dataBlockingStencil = FrameSpec{numChunks, chunkSize};
    auto longestSide = std::max(numNodesWithHalo.y(), numNodesWithHalo.x());
    auto dataBlockingBorder = FrameSpec{Vec{longestSide / chunkSize.x()}, Vec{std::max(chunkSize.y(), chunkSize.x())}};
    auto toRTime = FrameSpec{
        alpaka::Vec{dataBlockingStencil.m_numFrames.x(), dataBlockingStencil.m_numFrames.y()},
        Vec{dataBlockingStencil.m_frameExtent.x(), dataBlockingStencil.m_frameExtent.y()}};
    auto startTime_IN = std::chrono::high_resolution_clock::now();
    static auto tuningSession = getSessionFromExec(exec, toRTime, devAcc, numNodes);
    auto startTime_OUT = std::chrono::high_resolution_clock::now();
    auto ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(startTime_OUT - startTime_IN).count();
    std::cout << "[SessionInit]" << "," << ns_count << "," << "Stencil" << "\n";
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
        auto startTime_iN = std::chrono::high_resolution_clock::now();
        tuningSession.enqueue(
            devAcc,
            computeQueue,
            exec,
            toRTime,
            KernelBundle{stencilKernel, uCurrBufAcc.getMdSpan(), uNextBufAcc.getMdSpan(), numNodes, dx, dy, dt});
        auto endTime_IN = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsedTime_IN = endTime_IN - startTime_iN;
        auto ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime_IN - startTime_iN).count();
        std::cout << "[PHASE]" << "," << alpaka::tune::benchmark::phaseAccessor() << "," << "Stencil" << std::endl;
        log_event();
        // Apply boundaries
        computeQueue.enqueue(
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
        if(abortIfFinished(tuningSession))
            return 0;
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

auto main(int argc, char** argv) -> int
{
    using namespace alpaka;
    uint32_t i = 0;

    // Simple CLI parsing
    for(int arg = 1; arg < argc; ++arg)
    {
        std::string s = argv[arg];
        if(s.rfind("--numNodes=", 0) == 0)
        {
            i = static_cast<uint32_t>(std::stoul(s.substr(11)));
        }
    }
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
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, i); },
        onHost::allBackends(onHost::enabledApis));
};
