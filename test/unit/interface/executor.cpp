/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

struct GetExecutorKernel
{
    template<onAcc::concepts::Acc T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto result) const
    {
        auto executor = alpaka::getExecutor(acc);

        static_assert(concepts::Executor<ALPAKA_TYPEOF(executor)>);
        static_assert(std::is_same_v<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(acc.getExecutor())>);

        result[0] = executor == acc.getExecutor();
    }
};

TEMPLATE_LIST_TEST_CASE("get executor", "[interface][executor]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto executor = alpaka::getExecutor(cfg);

    static_assert(concepts::HasExecutor<ALPAKA_TYPEOF(cfg)>);
    static_assert(concepts::Executor<ALPAKA_TYPEOF(executor)>);
    CHECK((executor == cfg[object::exec]));

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        SKIP("No device available for " << deviceSpec.getName());
    }

    onHost::Device device = devSelector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});
    auto hBuff = onHost::allocHostLike(dBuff);

    queue.enqueue(executor, onHost::FrameSpec{Vec{1u}, Vec{1u}}, KernelBundle{GetExecutorKernel{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    CHECK(hBuff[0]);
}
