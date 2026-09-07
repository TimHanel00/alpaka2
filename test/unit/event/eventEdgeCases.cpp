/* Copyright 2026 alpaka contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <exception>
#include <thread>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("recorded event compared with itself has zero duration", "[queue-regression]", TestApis)
{
    auto device = test::getDeviceOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto event = queue.makeEvent();
    auto alias = event;
    queue.enqueue(event);

    // Both handles refer to one recording. The query must wait before returning zero.
    CHECK(onHost::getElapsedTime(event, alias) == std::chrono::duration<double>::zero());
    CHECK(event.isComplete());
}

TEMPLATE_LIST_TEST_CASE("elapsed time rejects unrecorded event operands", "[queue-regression]", TestApis)
{
    auto device = test::getDeviceOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto recorded = queue.makeEvent();
    auto unrecorded = queue.makeEvent();
    queue.enqueue(recorded);
    onHost::wait(recorded);

    // Only one operand has a timestamp. Require rejection in either argument
    // position, including the unrecorded self-comparison shortcut.
    CHECK_THROWS_AS(onHost::getElapsedTime(unrecorded, recorded), std::exception);
    CHECK_THROWS_AS(onHost::getElapsedTime(recorded, unrecorded), std::exception);
    CHECK_THROWS_AS(onHost::getElapsedTime(unrecorded, unrecorded), std::exception);
}

TEMPLATE_LIST_TEST_CASE("elapsed time uses the latest completed event recording", "[queue-regression]", TestApis)
{
    auto device = test::getDeviceOrSkipTest(TestType::makeDict());
    auto queue = device.makeQueue(queueKind::nonBlocking, timing::enabled);
    auto start = queue.makeEvent();
    auto end = queue.makeEvent();

    // First recording: start -> delay -> end, so end - start must be positive.
    queue.enqueue(start);
    // Separate the markers so an unchanged timestamp cannot pass through zero.
    queue.enqueueHostFn([] { std::this_thread::sleep_for(std::chrono::milliseconds{10}); });
    queue.enqueue(end);
    onHost::wait(queue);
    CHECK(onHost::getElapsedTime(start, end) > std::chrono::duration<double>::zero());

    // Re-recording: end -> delay -> new start, so end - start must now be negative.
    queue.enqueueHostFn([] { std::this_thread::sleep_for(std::chrono::milliseconds{10}); });
    queue.enqueue(start);
    onHost::wait(queue);
    CHECK(onHost::getElapsedTime(start, end) < std::chrono::duration<double>::zero());
}
