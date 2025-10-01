
#include "babelStreamCommon.hpp"
#include "catch2/catch_session.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include "DotKernelTuneableTrait.hpp"
using namespace alpaka;

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot. NStream is
 * optional. Init kernel is run before 5 standard kernel sequence. Babelstream is a memory-bound benchmark since the
 * main operation in the kernels has high Code Balance (bytes/FLOP) value. For example c[i] = a[i] + b[i]; has 2 reads
 * 1 writes and has one FLOP operation. For double precision each read-write is 8 bytes. Hence Code Balance (3*8 / 1) =
 * 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments and for kernels: init, copy, mul, add, triad (and dot kernel if a multi-thread acc
 * available):
 * ./babelstream --array-size=33554432 --number-runs=100
 * Run with Custom arguments and select from 3 kernel groups: all, triad, nstream
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=triad (only triad kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=nstream (only nstream kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=all (default case. Add, Multiply, Copy, Triad
 * and Dot) Run with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and default array size and num runs:
 * ./babelstream --success
 * ./babelstream -r xml
 * Run with Custom and catch2 arguments together:
 * ./babelstream  --success --array-size=1280000 --number-runs=10
 * Help to list custom and catch2 arguments
 * ./babelstream -?
 * ./babelstream --help
 *  According to tests, 2^25 or larger data size values are needed for proper benchmarking:
 *  ./babelstream --array-size=33554432 --number-runs=100
 */

// Main function that integrates Catch2 and custom argument handling
int main(int argc, char* argv[])
{
    // Handle custom arguments
    handleCustomArguments(argc, argv);

    // Initialize Catch2 and pass the command-line arguments to it
    int result = Catch::Session().run(argc, argv);

    // Return the result of the tests
    return result;
}
template<typename T, typename CVec>
consteval auto aligning() {
    return std::bit_ceil(sizeof(T) * CVec{}[0]);
}
struct SimdForEachKernel
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::MdSpan auto const& arg0,
        alpaka::concepts::MdSpan auto const&... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};

        simdGrid.concurrent(acc, arg0.getExtents(), func, arg0, args...);
    }
};
template <typename CVec,typename Data>
struct SimdForEachKernel_Add
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::MdSpan auto const& arg0,
        alpaka::concepts::MdSpan auto const&... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        CVec constexpr vec=CVec{};
        auto constexpr firstElem=vec.x();

        auto constexpr simdBytes=aligning<Data,CVec>();
        simdGrid.concurrent<simdBytes>(acc, arg0.getExtents(), func, arg0, args...);
    }
};
template <typename CVec,typename Data>
struct SimdForEachKernel_Mult
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::MdSpan auto const& arg0,
        alpaka::concepts::MdSpan auto const&... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        CVec constexpr vec=CVec{};
        auto constexpr firstElem=vec.x();

        auto constexpr simdBytes=aligning<Data,CVec>();
        simdGrid.concurrent<simdBytes>(acc, arg0.getExtents(), func, arg0, args...);
    }
};
template <typename CVec,typename Data>
struct SimdForEachKernel_Copy
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::MdSpan auto const& arg0,
        alpaka::concepts::MdSpan auto const&... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        CVec constexpr vec=CVec{};
        auto constexpr firstElem=vec.x();

        auto constexpr simdBytes=aligning<Data,CVec>();
        simdGrid.concurrent<simdBytes>(acc, arg0.getExtents(), func, arg0, args...);
    }
};
template <typename CVec,typename Data>
struct SimdForEachKernel_Triad
{
    //! \param acc The accelerator to be executed on.
    //! \param func functor applied to each SIMD package.
    //! \param arg0 MdSpan from which the problem size is derived
    //! \param args MdSpan other spans
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto const& func,
        alpaka::concepts::MdSpan auto const& arg0,
        alpaka::concepts::MdSpan auto const&... args) const
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        CVec constexpr vec=CVec{};
        auto constexpr firstElem=vec.x();

        auto constexpr simdBytes=aligning<Data,CVec>();
        simdGrid.concurrent<simdBytes>(acc, arg0.getExtents(), func, arg0, args...);
    }
};
struct SimdInitOp
{
    constexpr void operator()(auto const&, auto a, auto b, auto c) const
    {
        using SimdType = ALPAKA_TYPEOF(a.load());
        a = SimdType::all(initA);
        b = SimdType::all(initB);
        c = SimdType::all(initC);
    }
};

struct SimdCopyOp
{
    constexpr void operator()(auto const&, auto const a, auto c) const
    {
        c = a.load();
    }
};

struct SimdMultOp
{
    constexpr void operator()(auto const&, auto b, auto const c) const
    {
        using T = trait::GetValueType_t<ALPAKA_TYPEOF(b)>;
        T const scalar = static_cast<T>(scalarVal);
        b = scalar * c.load();
    }
};

struct SimdAddOp
{
    constexpr void operator()(auto const&, auto const a, auto b, auto c) const
    {
        c = a.load() + b.load();
    }
};

struct SimdTriadOp
{
    constexpr void operator()(auto const&, auto a, auto const b, auto const c) const
    {
        using T = trait::GetValueType_t<ALPAKA_TYPEOF(a)>;
        T const scalar = static_cast<T>(scalarVal);
        a = b.load() + scalar * c.load();
    }
};

struct SimdNStreamOp
{
    constexpr void operator()(auto const&, auto a, auto const b, auto const c) const
    {
        using T = trait::GetValueType_t<ALPAKA_TYPEOF(a)>;
        T const scalar = static_cast<T>(scalarVal);
        a = a.load() + b.load() + scalar * c.load();
    }
};
template<typename T, typename CVec>
inline constexpr std::uint32_t SimdBytes = CVec{}.x() * sizeof(T);
//! Dot product of two vectors. The result is not a scalar but a vector of block-level dot products. For the
//! BabelStream implementation and documentation: https://github.com/UoB-HPC
template<typename CVec,typename Data>
struct DotKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a MdSpan for vector a
    //! \param b MdSpan for vector b
    //! \param sum Pointer for result vector consisting sums of blocks
    //! \param arraySize the size of the array
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const a,
        alpaka::concepts::MdSpan auto const b,
        auto sum,
        auto arraySize) const
    {
        using T = trait::GetValueType_t<ALPAKA_TYPEOF(sum)>;
        auto sdata = onAcc::getDynSharedMem<T>(acc);

        std::uint32_t constexpr simdBytes=aligning<Data,CVec>();

        auto frameExtent = acc[frame::extent];
        auto numElemsPerFrame=CVec{}[0]*frameExtent[0];
        auto numFrames=arraySize/numElemsPerFrame;
        auto tbSum = alpaka::makeMdSpan(
            sdata,
            frameExtent,
            alpaka::calculatePitchesFromExtents<T>(frameExtent),
            Alignment{});
        //auto tbSum = onAcc::declareSharedMdArray<T, uniqueId()>(acc, CVec<uint32_t, blockThreadExtentMain>{});
#if 1


        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});
        /* init shared memory
         * no need to synchronize because after the loop because the same index map will be used to access the shared
         * memory in the tribble loop.
         */
        for(auto [elemIdxInFrame] : traverseInFrame)
        {
            tbSum[elemIdxInFrame] = T{0};
        }

        auto const frameDataExtent = numFrames * frameExtent;
        auto traverseOverFrames = onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{alpaka::Vec{0u}, frameDataExtent, frameExtent});

        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdxInFrame : traverseInFrame)
            {
                auto allThreads = onAcc::SimdAlgo{onAcc::WorkerGroup{frameIdx + elemIdxInFrame, frameDataExtent}};
                auto reducedValue = allThreads.template transformReduce<simdBytes>(
                    acc,
                    alpaka::Vec{arraySize},
                    T{0},
                    std::plus{},
                    [&](auto const&, auto&& simdA, auto&& simdB) constexpr { return simdA.load() * simdB.load(); },
                    a,
                    b);

                tbSum[elemIdxInFrame] += reducedValue;
            }
        }
        // sync is required because we do not know which thread wrote whcih value
        alpaka::onAcc::syncBlockThreads(acc);
        // aggregate for each thread but skip the first shared memory slot
        for(auto [elemIdxInFrame] :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(), frameExtent}))
        {
            tbSum[acc[layer::thread].idx()] += tbSum[elemIdxInFrame];
        }

#else
        // this version is shorter in code but runs into floating point stability issues due to to long aggregation of
        // value by a single thread
        auto threadSum = T{0};
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
        {
            threadSum += a[i] * b[i];
        }
        for(auto [local_i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::threadsInBlock))
        {
            tbSum[local_i] = threadSum;
        }
#endif
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            alpaka::onAcc::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }
        if(local_i == 0)
            onAcc::atomicAdd(acc, &sum[0], tbSum[local_i]);
    }
};
constexpr bool isPowerOfTwo(std::size_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}
template<typename T_1, typename T_2>
struct Sessions{
    T_1 &DotSession;
    T_2 &RestSession;
    Sessions(T_1 &DotSession, T_2 &RestSession):DotSession(DotSession), RestSession(RestSession){};
};

template<typename Data_T,typename Exec_T>
static auto getSessionFromExec(Exec_T const &exec,auto arraySize,auto & devAcc){

		//default for GPUs
        using Idx = std::uint32_t;
    using idxVec = alpaka::Vec<uint32_t, 1u>;
        std::string data;
        if(sizeof(Data_T)==sizeof(float)){
            data="float";
           }
          else{
              data="double";
              }
    auto setFixedNumBlocks_ = devAcc.getDeviceProperties().m_multiProcessorCount;
    auto maxThreads = devAcc.getDeviceProperties().m_maxThreadsPerBlock;
    std::cout << "max Threads" << maxThreads << std::endl;
    auto mpVec = idxVec{static_cast<Idx>(setFixedNumBlocks_) * static_cast<Idx>(8u)};
    static constexpr auto _0T = std::size_t{0};
    // alpaka::tune::Tuneable{uVec{56*2}, IdxRange{uVec{56*2}, uVec{dataBlocking.m_numFrames}, uVec{56*2}}})
    static auto tuningSessionDot
        = tune::TuningBuilder{}
              .withRunSpecifiers(std::to_string(arraySize))
              .withFrameExtentTune(tune::Tuneable(IdxRange{idxVec{64}, idxVec{64 * 16}, idxVec{64}}))
              .withBlockSizeTune(tune::Tuneable(IdxRange{idxVec{64}, idxVec{maxThreads}, idxVec{64}}))
              .withNumBlocksTune()
              .template withConstraint<tune::frameTune::FrameExtent, _0T>(
                  [arraySize](auto frameExtent, auto concurrentElements)
                  {
                      auto chunkElements = concurrentElements * frameExtent;
                      auto condZ = arraySize % chunkElements[0] == decltype(arraySize){0};

                      return condZ&&isPowerOfTwo(concurrentElements[0]);
                  })
              .withConfig("./config/realBabelstreamGPU_"+std::to_string(arraySize)+"_.toml")
              .build();
    static auto tuningSessionRest
        = tune::TuningBuilder{}
              .withRunSpecifiers(std::to_string(arraySize))
              .withBlockSizeTune(tune::Tuneable(IdxRange{idxVec{64}, idxVec{maxThreads}, idxVec{64}}))
              .withNumBlocksTune()
              .template withConstraint<_0T>(  [arraySize](auto concurrentElements){return isPowerOfTwo(concurrentElements[0]);})
              .withConfig("./config/realBabelstreamGPU_Rest_"+std::to_string(arraySize)+"_.toml")
              .build();
		return Sessions(tuningSessionDot,tuningSessionRest);
    };
template<typename T_TuningSessionDot,typename T_TuningSessionRest>
bool abortIfFinished(const T_TuningSessionDot &dotSession,const  T_TuningSessionRest &restSession){
    #ifdef Debug
    std::cout<<" sessions Finished rest: "<<restSession.finishedConfigs<<std::endl;
    std::cout<<" sessions Finished dot: "<<dotSession.finishedConfigs<<std::endl;
    #endif
    if(restSession.finishedConfigs>=4&&dotSession.finishedConfigs>=1){return true;};
    return false;
    }
template<typename Data_T>
static auto getSessionFromExec(alpaka::exec::CpuOmpBlocks const &exec, auto arraySize, auto &devAcc) {
    // specialization implementation
using Idx = std::uint32_t;
    using idxVec = alpaka::Vec<uint32_t, 1u>;
 std::string data;
        if(sizeof(Data_T)==sizeof(float)){
            data="float";
           }
          else{
              data="double";
              }
	std::cout<<" selected for cpu omp blocks"<<std::endl;
	const auto setFixedNumBlocks_ = devAcc.getDeviceProperties().m_multiProcessorCount;
    auto mpVec = idxVec{static_cast<Idx>(setFixedNumBlocks_)};
    static auto constexpr _0T=static_cast<std::size_t>(0);
	static auto sessionDot=tune::TuningBuilder{}
              .withRunSpecifiers(std::to_string(arraySize),data)
              .withFrameExtentTune(tune::Tuneable(IdxRange{idxVec{64}, idxVec{64 * 16}, idxVec{64}}))
                            .template withConstraint< tune::frameTune::FrameExtent, _0T>(
                  [arraySize]( auto frameExtent, auto concurrentElements)
                  {
                      auto chunkElements = concurrentElements * frameExtent;
                      auto condZ = arraySize % chunkElements[0] == decltype(arraySize){0};

                      return condZ&&isPowerOfTwo(concurrentElements[0]);
                  })
              .withNumBlocksTune()
    .withConfig("./config/Babelstream_CPU_"+std::to_string(arraySize)+"_.toml").build();
	static auto sessionRest= tune::TuningBuilder{}
          .withRunSpecifiers(std::to_string(arraySize),data)
          .withNumBlocksTune()
          .template withConstraint<_0T>(  [arraySize](auto concurrentElements){return isPowerOfTwo(concurrentElements[0]);})
          .withConfig("./config/Babelstream_CPU_"+std::to_string(arraySize)+"_.toml")
          .build();

		return Sessions(sessionDot,sessionRest);
}
void log_event(const std::string& label) {
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count() % 1'000'000'000;

    std::tm* tm = std::localtime(&t_c);
    std::cout << "[" << std::put_time(tm, "%F %T") << "." << std::setfill('0') << std::setw(9) << ns
              << "]," << label << std::endl;
}
//! \brief The Function for testing babelstream kernels for given Acc type and data type.
//! \tparam TAcc the accelerator type
//! \tparam DataType The data type to differentiate single or double data type based tests.
template<typename DataType>
void testKernels(auto const deviceSpec, auto const exec)
{
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        std::cout << "Kernels: Init, Copy, Mul, Add, Triad, Dot Kernels" << std::endl;
    }

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device devAcc = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
    // support for double precision is not guaranteed for sycl devices such as Intel GPUs
    if constexpr(std::is_same_v<DataType, double> && std::is_same_v<decltype(deviceSpec.getApi()), api::OneApi>)
    {
        if(devAcc.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size() == 0)
        {
            std::cout << onHost::getName(devAcc) << " does not support double precision"
                      << "\n";
            std::cout << "Skip benchmark.\n";
            std::cout << "For Intel Arc GPUs, use the environemnt variables `IGC_EnableDPEmulation=1 "
                         "wDefaultFP64Settings=1` to emulate double precision support.\n";
            return;
        }
    }
#endif

    std::cout << devAcc.getDeviceProperties() << std::endl;

    std::cout << "used exec " << onHost::demangledName(exec) << std::endl;

    // A MetaData class instance to keep the benchmark info and results to print later. Does not include intermediate
    // runtime data.
    BenchmarkMetaData metaData;

    // Convert data-type to string to display
    std::string dataTypeStr;
    if(std::is_same<DataType, float>::value)
    {
        dataTypeStr = "single";
    }
    else if(std::is_same<DataType, double>::value)
    {
        return;
        dataTypeStr = "double";
    }

    // Get the host device for allocating memory on the host
    onHost::Queue queue = devAcc.makeQueue();

    // Create vectors
    using Idx = std::uint32_t;
    using idxVec = alpaka::Vec<uint32_t, 1u>;
    auto arraySize = static_cast<Idx>(arraySizeMain);

    // Acc buffers
    auto bufAccInputA = onHost::alloc<DataType>(devAcc, arraySize);
    auto bufAccInputB = onHost::allocLike(devAcc, bufAccInputA);
    auto bufAccOutputC = onHost::allocLike(devAcc, bufAccInputA);

    // Host buffer as the result
    auto bufHostOutputA = onHost::allocHostLike(bufAccInputA);
    auto bufHostOutputB = onHost::allocHostLike(bufAccInputB);
    auto bufHostOutputC = onHost::allocHostLike(bufAccOutputC);

    /* Each frame will have 64 elements processed by each thread.
     * The number of frames is calculated based on the array size and the number of elements processed by each thread.
     *
     * @todo The value is currently a magic number but should be derived from the SIMD width of the device and a factor
     * to reflect the instruction level parallelism. This is currently not well abstracted in alpaka and requires that
     * a kernel can reflect the concurrency bytes used for the `SimdAlgo::concurrent()` back to the host, e.g. some
     * as we use for dynamic shared memory.
     */
    uint32_t elementsPerFrameItem = getNumElemPerThread<DataType>(queue);

	std::cout<<"ELEMENTSPERFRAME "<<elementsPerFrameItem<<std::endl;
    accessArraySize<idxVec>(arraySize);
    auto numFramesInit = arraySize/ (static_cast<Idx>(blockThreadExtentMain) * elementsPerFrameItem);
    auto dataBlockingInit=onHost::FrameSpec{
        idxVec{static_cast<Idx>(numFramesInit)},
        idxVec{static_cast<Idx>(blockThreadExtentMain)}};
    auto numFrames = arraySize/ (static_cast<Idx>(blockThreadExtentMain) * 1);
    auto dataBlocking = onHost::FrameSpec{
        idxVec{static_cast<Idx>(numFrames)},
        idxVec{static_cast<Idx>(blockThreadExtentMain)}};
    auto dataBlockingDot = onHost::FrameSpec{
        idxVec{static_cast<Idx>(numFrames)},
        idxVec{static_cast<Idx>(blockThreadExtentMain/2)}}; //restrict the dotKernell search space a bit
    // alpaka::tune::Tuneable{uVec{56*2}, IdxRange{uVec{56*2}, uVec{dataBlocking.m_numFrames}, uVec{56*2}}})
    auto tuningSessions
        = getSessionFromExec<DataType>(exec,arraySize,devAcc);

    auto& tuningSessionDot = tuningSessions.DotSession;
    auto &tuningSessionRest = tuningSessions.RestSession;


    // To record runtime data generated while running the kernels
    RuntimeResults runtimeResults;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        std::size_t nsec_count=kernelFunc();
        std::cout<<"[PHASE]"<< ","<<alpaka::tune::benchmark::phaseAccessor()<< "," << kernelLabel<<"\n";
        runtimeResults.kernelToRundataMap[kernelLabel]->timingsSuccessiveRuns.push_back(nsec_count);

		log_event(kernelLabel);

    };


    // Initialize logger before running kernels
    // Runtime result initialisation to be filled by each kernel
    runtimeResults.addKernelTimingsVec("InitKernel");
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        runtimeResults.addKernelTimingsVec("CopyKernel");
        runtimeResults.addKernelTimingsVec("AddKernel");
        runtimeResults.addKernelTimingsVec("TriadKernel");
        runtimeResults.addKernelTimingsVec("MultKernel");
        runtimeResults.addKernelTimingsVec("DotKernel");
    }
    if(kernelsToBeExecuted == KernelsToRun::Copy)
        runtimeResults.addKernelTimingsVec("CopyKernel");
    if(kernelsToBeExecuted == KernelsToRun::Mult)
        runtimeResults.addKernelTimingsVec("MultKernel");
    if(kernelsToBeExecuted == KernelsToRun::Add)
        runtimeResults.addKernelTimingsVec("AddKernel");
    if(kernelsToBeExecuted == KernelsToRun::Dot)
        runtimeResults.addKernelTimingsVec("DotKernel");
    if(kernelsToBeExecuted == KernelsToRun::Triad)
        runtimeResults.addKernelTimingsVec("TriadKernel");

    // Init kernel
    measureKernelExec(
        [&]()
        {
            onHost::wait(queue);
            auto start = std::chrono::high_resolution_clock::now();

            queue.enqueue(
                exec,
                dataBlockingInit,
                KernelBundle{SimdForEachKernel{}, SimdInitOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
            auto end = std::chrono::high_resolution_clock::now();
            return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        },
        "InitKernel");

    // Init kernel will be run for all cases therefore add it to metadata unconditionally
    metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);

    // Dot kernel result
    DataType resultDot = static_cast<DataType>(0.0);

    // Main for loop to run the kernel-sequence
    for(auto i = 0; i < numberOfRuns; i++)
    {
        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Copy)
        {
            // Test the copy-kernel. Copy A one by one to C.
            measureKernelExec(
                [&]() {
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    tuningSessionRest.enqueue(
                        queue,
                        exec,
                        dataBlocking,
                        KernelBundle{SimdForEachKernel_Copy<CVec<std::uint32_t, 1>, DataType>{}, SimdCopyOp{}, bufAccInputA, bufAccOutputC});
                    auto end = std::chrono::high_resolution_clock::now();
                    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                },
                "CopyKernel");
        }

        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Mult)
        {
            measureKernelExec(
                [&]() {
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    tuningSessionRest.enqueue(
                        queue,
                        exec,
                        dataBlocking,
                        KernelBundle{SimdForEachKernel_Mult<CVec<std::uint32_t, 1>, DataType>{}, SimdMultOp{}, bufAccInputB, bufAccOutputC});
                    auto end = std::chrono::high_resolution_clock::now();
                    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                },
                "MultKernel");
        }

        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Add)
        {
            measureKernelExec(
                [&]() {
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    tuningSessionRest.enqueue(
                        queue,
                        exec,
                        dataBlocking,
                        KernelBundle{SimdForEachKernel_Add<CVec<std::uint32_t, 1>, DataType>{}, SimdAddOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
                    auto end = std::chrono::high_resolution_clock::now();
                    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                },
                "AddKernel");
        }
                if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Triad)
        {
            measureKernelExec(
                [&]() {
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    tuningSessionRest.enqueue(
                        queue,
                        exec,
                        dataBlocking,
                        KernelBundle{SimdForEachKernel_Triad<CVec<std::uint32_t, 1>, DataType>{}, SimdTriadOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});

                    auto end = std::chrono::high_resolution_clock::now();

                    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                },
                "TriadKernel");
        }

        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Dot)

        {
            uint32_t elementsPerFrameItem = getNumElemPerThread<DataType>(queue);
            auto numFrames = std::min(
                static_cast<Idx>(dotGridBlockExtent),
                alpaka::divExZero(arraySize, (static_cast<Idx>(blockThreadExtentMain) * elementsPerFrameItem)));

            auto dataBlockingDot = onHost::FrameSpec{numFrames, static_cast<Idx>(blockThreadExtentMain)};

            // Vector of sums of each block
            auto bufAccSumPerBlock = onHost::alloc<DataType>(devAcc, 1u);
            auto bufHostSumPerBlock = onHost::allocHostLike(bufAccSumPerBlock);

            measureKernelExec(
                [&]() {
                    // the memset and copy from acc operations impose additional overhead which makes
                    // it unfeasible to measure the raw tuner overhead
                    onHost::memset(queue, bufAccSumPerBlock, 0);
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    tuningSessionDot.enqueue(
                        queue,
                        exec,
                        dataBlockingDot,
                        KernelBundle{
                            DotKernel<CVec<std::uint32_t, 1>, DataType>{},
                            bufAccInputA,
                            bufAccInputB,
                            bufAccSumPerBlock,
                            arraySize});
                    auto end = std::chrono::high_resolution_clock::now();
                    onHost::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock);
                    onHost::wait(queue);
                    resultDot = bufHostSumPerBlock[0u];
    				return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        },
        "DotKernel");
            // Add workdiv to the list of workdivs to print later
            metaData.setItem(BMInfoDataType::WorkDivDot, dataBlockingDot);
            }
        // NStream kernel is run only for one command line argument
        if(kernelsToBeExecuted == KernelsToRun::NStream)
        {
            // Test the NStream-kernel. Calculate A += B + scalar * C;
            measureKernelExec(
                [&]()
                {
                    onHost::wait(queue);
                    auto start = std::chrono::high_resolution_clock::now();

                    queue.enqueue(
                        exec,
                        dataBlocking,
                        KernelBundle{SimdForEachKernel{}, SimdNStreamOp{}, bufAccInputA, bufAccInputB, bufAccOutputC});
                    auto end = std::chrono::high_resolution_clock::now();
                    return static_cast<std::size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                },
                "NStreamKernel");
        }
        onHost::wait(queue);
        if(abortIfFinished(tuningSessionDot,tuningSessionRest))return;
	}

    // Copy results back to the host, measure copy time
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Copy arrays back to host since the execution of kernels except dot kernel finished
        onHost::memcpy(queue, bufHostOutputC, bufAccOutputC);
        onHost::memcpy(queue, bufHostOutputB, bufAccInputB);
        onHost::memcpy(queue, bufHostOutputA, bufAccInputA);
        onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // Get duration in seconds
        std::chrono::duration<double> duration = end - start;
        double copyRuntime = duration.count();
        metaData.setItem(BMInfoDataType::CopyTimeFromAccToHost, copyRuntime);
    }

    //
    // Result Verification and BW Calculation for 3 cases
    //

    // Generated expected values by doing the same chain of operations due to floating point error
    DataType expectedA = static_cast<DataType>(initA);
    DataType expectedB = static_cast<DataType>(initB);
    DataType expectedC = static_cast<DataType>(initC);

    // To calculate expected results by applying at host the same operation sequence
    calculateBabelstreamExpectedResults(expectedA, expectedB, expectedC);

    // Verify the resulting data, if kernels are init, copy, mul, add, triad and dot kernel
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        // Find sum of the errors as sum of the differences from expected values
        constexpr DataType initVal{static_cast<DataType>(0.0)};
        DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrC += std::fabs(std::data(bufHostOutputC)[static_cast<Idx>(i)] - expectedC);
            sumErrB += std::fabs(std::data(bufHostOutputB)[static_cast<Idx>(i)] - expectedB);
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        // Normalize and compare sum of the errors
        // Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(sumErrC / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrB / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize), static_cast<DataType>(0.0)));
        onHost::wait(queue);

        // Verify Dot kernel
        DataType const expectedSum = static_cast<DataType>(arraySize) * expectedA * expectedB;
        //  Dot product should be identical to arraySize*valA*valB
        //  Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(static_cast<float>(std::fabs(resultDot - expectedSum) / expectedSum), 0.0f));

        // Set workdivs of benchmark metadata to be displayed at the end
        metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivCopy, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivAdd, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivMult, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the Triad Kernel result if "--run-kernels=triad".
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        // Verify triad by summing the error
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize) / expectedA, static_cast<DataType>(0.0)));
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the NStream Kernel result if "--run-kernels=nstream".
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize) / expectedA, static_cast<DataType>(0.0)));

        metaData.setItem(BMInfoDataType::WorkDivNStream, dataBlocking);
    }

    // Runtime results of the benchmark: Calculate throughput and bandwidth
    // Set throuput values depending on the kernels
    runtimeResults.initializeByteReadWrite<DataType>(arraySize);
    runtimeResults.calculateBandwidthsForKernels<DataType>();

    // Set metadata to display all benchmark related information.
    //
    // All information about benchmark and results are stored in a single map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);
    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, onHost::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, onHost::demangledName(exec));
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");

    // get labels from the map
    std::vector<std::string> kernelLabels;
    std::transform(
        runtimeResults.kernelToRundataMap.begin(),
        runtimeResults.kernelToRundataMap.end(),
        std::back_inserter(kernelLabels),
        [](auto const& pair) { return pair.first; });
    // Join elements and create a comma separated string and set item
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> values(runtimeResults.getThroughputKernelArray());
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(values, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> valuesBW(runtimeResults.getBandwidthKernelVec());
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(valuesBW, ", "));

    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(runtimeResults.getMinExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(runtimeResults.getMaxExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(runtimeResults.getAvgExecTimeKernelArray(), ", "));
    // Print the summary as a table, if a standard serialization is needed other functions of the class can be used
    std::cout << metaData.serializeAsTable() << std::endl;
}

using Backends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;
// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Float>", "[benchmark-test]", Backends)
{
    auto backend = TestType::makeDict();
    // Run tests for the float data type
    testKernels<float>(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]);
}
/*
// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Double>", "[benchmark-test]", Backends)
{
    auto backend = TestType::makeDict();
    // Run tests for the double data type
    testKernels<float>(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]);
}*/
