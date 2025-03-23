//
// Created by tim on 27.01.25.
//

#ifndef STARTKERNELWRAPPER_HPP
#define STARTKERNELWRAPPER_HPP
#include "reduceKernel.hpp"

static std::size_t getNumberRuns()
{
    if(char const* var = std::getenv("number_runs"))
    {
        try
        {
            std::size_t const value = static_cast<std::size_t>(std::stoul(var));
            return value;
        }
        catch(std::exception const& e)
        {
            std::cerr << "Invalid value for number_runs: " << e.what() << std::endl;
        }
    }
    return 0;
}

/*
 * rounds a 1Dim integerType Vector to the closest n aligend Nr
 */
template<typename Vec, typename N>
auto ALPAKA_FN_HOST_ACC roundAlign(Vec vec, N n)
{
    return n * (vec / n);
}

template<
    std::size_t numLoads = 4,
    std::size_t stride = 4,
    typename operationType,
    typename Exec,
    typename DevHost,
    typename DevAcc,
    typename BufType>
auto reduction(operationType const& type, Exec& exec, DevHost& devHost, DevAcc& devAcc, BufType& bufHost)
{
    using namespace alpaka;

    // type and variable definitions
    using IdxType = std::remove_reference_t<decltype(bufHost.getExtents().x())>;
    using IdxVec = std::remove_reference_t<decltype(bufHost.getExtents())>;
    constexpr IdxType nr_of_threads = 256;
    auto firstIndex = alpaka::mapToND<IdxType>(bufHost.getExtents(), static_cast<IdxType>(0));
    using T = std::remove_reference_t<decltype(bufHost[firstIndex])>;
    if(bufHost.getExtents().product() == 0)
    {
        throw std::runtime_error("Error: Can not reduce an empty Buffer!");
    }
    alpaka::Vec<std::size_t, 5>();
    onHost::Queue queue = devAcc.makeQueue();
    // define destination Buffer Size
    IdxVec destinationExtent = IdxVec{1};
    // allocate devide Memory for input Buffer
    auto bufAccA = alpaka::onHost::allocMirror(devAcc, bufHost);
    // allocate Memory for destination Buffer on the Host Side
    auto destHost = onHost::alloc<T>(devHost, destinationExtent);
    // fill destination Buffer with the neutral Element
    destHost[firstIndex] = reduce::neutral_element<T>(type);
    // allocate device Memory for desination Buffer
    auto destBuf = alpaka::onHost::allocMirror(devAcc, destHost);
    // copy the data of both buffers
    onHost::memcpy(queue, bufAccA, bufHost);
    onHost::memcpy(queue, destBuf, destHost);
    // wait until memory transfers are complete

    // how many frame cells do we need (numLods*stride elements are calculated at once = in one Cell)
    auto nr_of_FrameCells = IdxVec{bufHost.getExtents().x() / (stride * numLoads)};
    // handle edge cases where the number of Elements is too small for one FrameExtent
    auto alterFrameExtent = IdxVec{std::min(nr_of_FrameCells.x(), nr_of_threads)};
    auto trueFrameExtent = IdxVec{nr_of_threads};
    while(trueFrameExtent > nr_of_FrameCells.x())
    {
        trueFrameExtent /= IdxVec{2};
    }
    // numFrames = performs a divFloor on the number of frame cells ~ how many frame extends are completely covered
    // with data
    auto const numFrames = std::max(IdxVec{1u}, nr_of_FrameCells / trueFrameExtent);
    // the first Kernel will be given this Frame Spec
    auto firstKernelFrame = onHost::FrameSpec{numFrames, trueFrameExtent};
    // whats the most data we can cover (calculate the End of the Range) we can cover with the current frame Spec
    auto const VecFirstExtent
        = IdxVec{roundAlign(bufHost.getExtents(), numFrames * trueFrameExtent.product() * stride * numLoads).x()};
    auto const remElements = bufHost.getExtents() - VecFirstExtent;
    // since for the last Kernel we use a stride size of 1 and a number of loads = 1 we dont have to worry about
    // alignment
    auto frameExtentTmp = trueFrameExtent;
    // reducing the number of frames~threads spawned for the final Kernel to make sure we have a lower number of idle
    // threads
    while(frameExtentTmp.product() > remElements.product())
    {
        frameExtentTmp /= IdxVec{2};
    }
    // std::cout<<"frameExentTMP: "<<frameExtentTmp.x()<<std::endl;
    // use only 1 Block for the final Kernel, if frameExtent<remElements we ensure that all elements are covered with a
    // frameSpec-strided inner loop
    auto lastKernelFrame = onHost::FrameSpec{
        static_cast<IdxType>(1),
        static_cast<IdxType>(frameExtentTmp),
        static_cast<IdxType>(alterFrameExtent)};
    // Instantiate the kernel object with its respective dynamic shared memory given inside {}
    Reduce<stride, numLoads, T> kernel1{static_cast<uint32_t>(trueFrameExtent.x() * sizeof(T))};
    // Instantiate the final Kernel object with the maximum shared memory it may require
    Reduce<IdxType{1}, IdxType{1}, T> kernel2{static_cast<uint32_t>(frameExtentTmp.x() * sizeof(T))};
    // with the data transfers to device completed we can now instantiate our Kernel Bundles (i.e. specify parameters
    auto taskKernel = KernelBundle{
        kernel1,
        type,
        bufAccA.getMdSpan(),
        destBuf.getMdSpan(),
        IdxVec{0},
        VecFirstExtent}; // this causes errors.
    using Vec1 = alpaka::Vec<std::size_t, 1u>;
    //.registerCompileTime(Reduce<stride,numLoads,T>::dynSharedMemBytes);
    std::cout << " Device: from Outer: " << typeid(ALPAKA_TYPEOF(devAcc)).name() << std::endl;
    TuningSession session{tune::strategy::randomSearch{}};
    auto latestSession = session.withBlockSizeTune(tune::ThreadBlockSizeTune{Vec{1}, IdxRange{Vec{1}, Vec{1}, Vec{1}}})
                             .withGridSizeTune(alpaka::Vec<std::size_t, 1u>{6})
                             .withRunSpecifiers(bufHost.getExtents().product())
                             .withConfig("./config/reduce.toml");
    auto const taskKernelLeftOver = KernelBundle{
        kernel2,
        type,
        bufAccA.getMdSpan(),
        destBuf.getMdSpan(),
        VecFirstExtent.x(),
        bufHost.getExtents(),
        alpaka::tune::Tuneable<std::size_t>(2)};
    onHost::wait(queue);

    for(auto i = 0; i < getNumberRuns(); i++)
    {
        // auto event=session.createTimeEvent();//creates a tuning event for the current kernel call will in unique
        // circumstance be used enqueue both Kernels queue ensures sequential execution
        latestSession.enqueue(devAcc, queue, exec, firstKernelFrame, taskKernel);
        onHost::wait(queue);
    }
    std::cout << alpaka::core::demangledName(exec) << std::endl;
    // copy back results
    onHost::memcpy(queue, destHost, destBuf, alpaka::Vec{static_cast<T>(1)});
    onHost::wait(queue);
    return destHost.getMdSpan()[firstIndex];
}

#endif // STARTKERNELWRAPPER_HPP
