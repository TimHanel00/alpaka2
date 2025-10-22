//
// Created by tim on 20.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tune/tunable/generators.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, onHost::example::enabledExecutors))>;
using V2u = Vec<uint32_t, 2u>;

struct ExampleKernelA
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto outputVec,
        alpaka::concepts::Vector auto maxNumFrames,
        alpaka::concepts::Vector auto maxFrameExtent,
        alpaka::tune::concepts::Integral auto maxUserVal,
        auto val /*userTunable*/) const
    {
        // STATIC_CHECK(std::is_same_v<decltype(val), int>);
        // auto frameExtent = acc[frame::extent];
        // STATIC_CHECK(std::is_same_v<decltype(frameExtent), V2u>);
        // uint32_t frameExtentIdx = frameExtent.y() * maxFrameExtent.x() + frameExtent.x();
        //
        // auto numFrames = acc[frame::count];
        // STATIC_CHECK(std::is_same_v<decltype(numFrames), V2u>);
        // uint32_t numFramesIdx = numFrames.y() * maxNumFrames.x() + numFrames.x();
        // uint32_t globalTuningIdx
        //     = numFramesIdx * (maxFrameExtent.product() * maxUserVal) + frameExtentIdx * maxUserVal + val;
        // outputVec[globalTuningIdx] = true;
        //  no-op: just needs to compile with the user tunable present
    }
};

void initializeBuffer()
{
}

bool validateBuffer(alpaka::concepts::IBuffer auto const& buffer)
{
    for(auto const val : buffer)
    {
        if(val == false)
            return false;
    }
    return true;
}

void initTuning()
{
}

TEMPLATE_LIST_TEST_CASE(
    "enqueue with shallow frame placeholders + user int tunable",
    "[FrameSpecTuningModel][enqueue][shallow+user]",
    TestApis)
{
    using namespace alpaka;
    using namespace alpaka::tune;

    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    auto exec = cfg[object::exec];
    // INIT TUNING
    // Tuning session
    auto session = alpaka::tune::TuningBuilder{}
                       .withStrategy(strategy::exhaustiveSearch{})
                       .withContextSpecifier("sess-shallow-user")
                       .buildSession();

    // Frame spec + SHALLOW placeholders for frame space (tuner decides)
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{1, 1}};
    std::vector<V2u> vec=alpaka::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_numFrames,
                                  V2u{1, 1});
    auto frameModel = FrameSpecTuningModel{spec}
                          .withNumFramesTune(
                             vec) // shallow placeholder -> let Tuner decide
                          .withFrameExtentTune(
                              alpaka::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_frameExtent,
                                  V2u{1, 1})); // shallow placeholder -> let Tuner decide


    // User-provided tunable (single-dim int space), passed to the kernel via KernelBundle
    Tunable<7001u, int> userTune({0, 1, 2, 3}, 2, "UserIntTune"); // 4 elem tuning space

    // awiod
    auto bufferExtent = alpaka::Vec{4 * 4 * 4};
    auto uBufHost = alpaka::onHost::allocHost<bool>(bufferExtent);
    onHost::fill(queue, uBufHost, false);
    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocLike(queue.getDevice(), uBufHost);
    alpaka::onHost::memcpy(queue, uCurrBufAcc, uBufHost);
    alpaka::onHost::wait(queue);

    ///
    // Kernel bundle includes the user tunable as an argument
    auto kernelBundle = KernelBundle{ExampleKernelA{}, uCurrBufAcc.getMdSpan(), V2u{2, 2}, V2u{2, 2}, 4, userTune};
    for(auto i = 0; i < 1000; i++) // sufficient number of tuning times
    {
        // queue.enqueue(exec, spec, kernelBundle);
        session.enqueue(queue, exec, frameModel, kernelBundle);
    }
    // alpaka::onHost::memcpy(queue, uBufHost, uCurrBufAcc);
    // alpaka::onHost::wait(queue);
    // REQUIRE(validateBuffer(uBufHost)); // validate wether we actually supplied all of the tuning space on the device
}
