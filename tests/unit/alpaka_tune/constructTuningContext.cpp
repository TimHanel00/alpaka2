//
// Created by tim on 20.10.25.
//
#include "alpaka/tune/core/sessionBuilder.hpp"

#include <alpaka/tune/core/tuningSession.hpp>

#include <catch2/catch_test_macros.hpp>
using namespace alpaka::tune;

struct alpakaDeviceDummy
{
};

struct alpakaExecDummy
{
};

struct alpakaQueueDummy
{
    alpakaQueueDummy() = default;

    static auto getDevice()
    {
        return alpakaDeviceDummy{};
    }
};

struct kernelBundleDummy
{
};

TEST_CASE("construct context from simple session", "[TuningContext][create]")
{
    using V2u = alpaka::Vec<uint32_t, 2u>;
    auto session = TuningBuilder{}.withContextSpecifier("ab").buildSession();
    auto spec = alpaka::onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};
    auto frameModel = FrameSpecTuningModel{spec}.withFrameExtentTune().withNumBlocksTune();
    auto alpakaQueueDummyInit = alpakaQueueDummy{};
    auto alpakaExecDummyInit = alpakaExecDummy{};
    auto kernelBundle = kernelBundleDummy{};
    auto environmentPtr = detail::internal::setup_enqueue(
        alpakaQueueDummyInit,
        alpakaExecDummyInit,
        frameModel,
        kernelBundle,
        session);
    auto& environment_state = environmentPtr->environmentState;
}
