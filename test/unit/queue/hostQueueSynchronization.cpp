/* Copyright 2026 alpaka contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/api/host/OmpCollectiveQueue.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>

using namespace alpaka;

namespace
{
    // A timeout here means the operation is correctly blocked by our gate.
    // CTest's separate timeout detects failure to finish after releasing it.
    // The started promises precede API entry; this delay only encourages the interleaving.
    constexpr auto blockingInterval = std::chrono::milliseconds{100};
} // namespace

TEST_CASE("host blocking event wait allows concurrent re-recording", "[queue][queue-regression]")
{
    // Producer: gate -> E1. Consumer: wait(E1) -> E2, using the same event object.
    // Re-recording must leave the existing wait dependent on E1, not E2.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto waiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    auto const waitStatus = waiter.wait_for(blockingInterval);

    std::promise<void> recordStarted;
    auto recorder = std::async(
        std::launch::async,
        [&]
        {
            recordStarted.set_value();
            consumer.enqueue(event);
        });
    recordStarted.get_future().wait();
    recorder.wait_for(blockingInterval);

    releaseProducer.set_value();
    // Releasing the gate must allow E1 -> consumer wait -> E2 to finish.
    // Regression: recorder holds the event lock while waiting for the consumer
    // lock; the consumer holds its lock while E1 needs the event lock to finish.
    waiter.get();
    recorder.get();
    onHost::wait(producer);
    CHECK(waitStatus == std::future_status::timeout);
    CHECK(event.isComplete());
}

TEST_CASE("host blocking event wait keeps the queue nonempty", "[queue][queue-regression]")
{
    // Producer: gate -> event. Consumer: wait(event), with no other work.
    // An unresolved dependency still counts as unfinished queue work. If waitFor
    // bypasses the busy flag, isEmpty() and wait(queue) incorrectly report completion.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto eventWaiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    auto const eventWaitStatus = eventWaiter.wait_for(blockingInterval);
    bool const emptyWhileWaiting = consumer.isEmpty();

    // Queue synchronization must stay blocked for the same dependency.
    auto queueWaiter = std::async(std::launch::async, [&] { onHost::wait(consumer); });
    auto const queueWaitStatus = queueWaiter.wait_for(blockingInterval);
    releaseProducer.set_value();
    eventWaiter.get();
    queueWaiter.get();
    onHost::wait(producer);

    CHECK(eventWaitStatus == std::future_status::timeout);
    CHECK_FALSE(emptyWhileWaiting);
    CHECK(queueWaitStatus == std::future_status::timeout);
}

TEST_CASE("host queue wait includes callback destruction", "[queue][queue-regression]")
{
    // This test requires wait(queue) to include destruction of queue-owned captures.
    // Hold the destructor open after the callback body has already returned.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);

    struct Cleanup
    {
        std::promise<void>& started;
        std::shared_future<void> release;

        ~Cleanup()
        {
            started.set_value();
            release.wait();
        }
    };

    std::promise<void> cleanupStarted;
    std::promise<void> releaseCleanup;
    std::promise<void> releaseTask;
    auto cleanup = std::shared_ptr<Cleanup>(new Cleanup{cleanupStarted, releaseCleanup.get_future().share()});
    auto taskGate = releaseTask.get_future().share();
    queue.enqueueHostFn([cleanup, taskGate] { taskGate.wait(); });
    // Leave the queued callback as the last owner before allowing its body to return.
    cleanup.reset();
    releaseTask.set_value();
    cleanupStarted.get_future().wait();

    // Popping the task before destroying its capture makes the queue look empty.
    // The empty-queue shortcut then lets wait(queue) return while cleanup is blocked.
    auto waiter = std::async(std::launch::async, [&] { onHost::wait(queue); });
    auto const waitStatus = waiter.wait_for(blockingInterval);
    releaseCleanup.set_value();
    waiter.get();

    CHECK(waitStatus == std::future_status::timeout);
}

TEST_CASE("host blocking task exception restores queue state", "[queue][queue-regression]")
{
    // Throwing between setting and clearing the busy flag must still restore idle state.
    // Otherwise isEmpty() fails here and the collective wait below spins forever.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::blocking);
    auto throwingTask = [] { throw std::runtime_error{"host task failure"}; };

    CHECK_THROWS_AS(queue.enqueueHostFn(throwingTask), std::runtime_error);
    CHECK(queue.isEmpty());

#if ALPAKA_OMP
    auto collective = device.makeQueue(queueKind::ompCollective);
    CHECK_THROWS_AS(collective.enqueueHostFn(throwingTask), std::runtime_error);
    // A stale parent execution flag makes this collective wait spin forever.
#    pragma omp parallel num_threads(2)
    {
        onHost::wait(collective);
    }
#endif
}

TEST_CASE("host nonblocking queue wait reports a failed callback", "[queue][queue-regression]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);
    queue.enqueueHostFn([] { throw std::runtime_error{"host task failure"}; });

    // Proposed error contract: synchronization reports a preceding callback failure.
    // The worker stores it in the task's future; discarding that future loses the error.
    CHECK_THROWS_AS(onHost::wait(queue), std::runtime_error);
}

TEST_CASE("host event wait reports a preceding failed callback", "[queue][queue-regression]")
{
    // Queue: throwing callback -> event. Under the proposed error contract, reaching
    // the marker must also report the earlier failure; its own task succeeding is insufficient.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);
    auto event = queue.makeEvent();
    queue.enqueueHostFn([] { throw std::runtime_error{"host task failure"}; });
    queue.enqueue(event);

    CHECK_THROWS_AS(onHost::wait(event), std::runtime_error);
}

TEST_CASE("host native function preserves queue order and receives its handle", "[queue][queue-regression]")
{
    auto device = onHost::makeHostDevice();
    auto checkQueue = [&](concepts::QueueKind auto kind)
    {
        auto queue = device.makeQueue(kind);
        auto const handle = queue.getNativeHandle();
        std::atomic<std::uint32_t> phase{0u};
        std::atomic<bool> nativeSawPredecessor{false};
        std::atomic<bool> successorSawNative{false};
        std::atomic<bool> correctHandle{false};

        // Queue: host writes 1 -> native observes 1, writes 2 -> host observes 2 -> done.
        queue.enqueueHostFn([&] { phase = 1; });
        queue.enqueueNativeFn(
            [&](auto nativeHandle)
            {
                nativeSawPredecessor = phase == 1;
                correctHandle = nativeHandle == handle;
                phase = 2;
            });
        queue.enqueueHostFn([&] { successorSawNative = phase == 2; });
        auto done = queue.makeEvent();
        queue.enqueue(done);
        onHost::wait(done);

        CHECK(nativeSawPredecessor.load());
        CHECK(successorSawNative.load());
        CHECK(correctHandle.load());
    };

    SECTION("blocking")
    {
        checkQueue(queueKind::blocking);
    }
    SECTION("nonblocking")
    {
        checkQueue(queueKind::nonBlocking);
    }
}
