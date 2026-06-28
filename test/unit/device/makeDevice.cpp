/* Copyright 2026 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("make device from platform", "")
{
    auto platform = onHost::internal::makePlatform(api::host, deviceKind::cpu);

    Device device = onHost::makeDevice(*platform.get(), 0);
    Device directDevice = onHost::makeDeviceSelector(api::host, deviceKind::cpu).makeDevice(0);

    INFO(device.getName() << " == " << directDevice.getName());
    CHECK(device.getNativeHandle() == directDevice.getNativeHandle());
}

TEST_CASE("make device from device selector", "")
{
    auto hostSelector = onHost::makeDeviceSelector(api::host, deviceKind::cpu);

    Device device = onHost::makeDevice(hostSelector, 0);
    Device directDevice = hostSelector.makeDevice(0);

    INFO(device.getName() << " == " << directDevice.getName());
    CHECK(device.getNativeHandle() == directDevice.getNativeHandle());
}

TEST_CASE("make device from device spec", "")
{
    auto deviceSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};

    Device device = onHost::makeDevice(deviceSpec);
    Device directDevice = onHost::makeDeviceSelector(deviceSpec).makeDevice(0);

    INFO(device.getName() << " == " << directDevice.getName());
    CHECK(device.getNativeHandle() == directDevice.getNativeHandle());
}

TEST_CASE("make device from backend", "")
{
    onHost::concepts::Backend auto backend = Dict{
        DictEntry{object::deviceSpec, onHost::DeviceSpec{api::host, deviceKind::cpu}},
        DictEntry{object::exec, exec::cpuSerial}};

    Device device = onHost::makeDevice(backend, 0);
    Device directDevice = onHost::makeDeviceSelector(api::host, deviceKind::cpu).makeDevice(0);

    INFO(device.getName() << " == " << directDevice.getName());
    CHECK(device.getNativeHandle() == directDevice.getNativeHandle());
}
