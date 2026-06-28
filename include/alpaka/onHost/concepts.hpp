/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

namespace alpaka::onHost
{
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct DeviceSpec;

    namespace internal::concepts
    {
        template<typename T>
        concept Device = requires(T device) {
            { alpaka::internal::GetName::Op<T>{}(device) } -> std::convertible_to<std::string>;
            { internal::MakeEvent::Op<T>{}(device) };
            { internal::GetNativeHandle::Op<T>{}(device) };
            { internal::GetDeviceProperties::Op<T>{}(device) };
        };

        template<typename T>
        concept Platform = requires(T platform) {
            { alpaka::internal::GetName::Op<T>{}(platform) };
        };

        template<typename T>
        concept Queue = requires(T device) {
            { alpaka::internal::GetName::Op<T>{}(device) } -> std::convertible_to<std::string>;
            { internal::GetNativeHandle::Op<T>{}(device) };
        };

        template<typename T>
        concept QueueHandle = requires(T t) {
            typename T::element_type;
            requires Queue<typename T::element_type>;
        };

        template<typename T>
        concept PlatformHandle = requires(T t) {
            typename T::element_type;
            requires Platform<typename T::element_type>;
        };

        template<typename T>
        concept DeviceHandle = requires(T t) {
            typename T::element_type;
            requires Device<typename T::element_type>;
        };

        template<typename T>
        using BackendDeviceSpec = std::remove_cvref_t<decltype(std::declval<T const&>()[alpaka::object::deviceSpec])>;

        template<typename T>
        using BackendExecutor = std::remove_cvref_t<decltype(std::declval<T const&>()[alpaka::object::exec])>;
    } // namespace internal::concepts

    namespace concepts
    {
        template<typename T>
        concept NameHandle = requires(T t) {
            typename T::element_type;
            requires alpaka::concepts::HasName<typename T::element_type>;
        };

        template<typename T>
        concept StaticNameHandle = requires(T t) {
            typename T::element_type;
            requires alpaka::concepts::HasStaticName<typename T::element_type>;
        };

        /** Dictionary describing a backend configuration.
         *
         * A backend is the combination of a device specification and an executor, as returned by
         * alpaka::onHost::allBackends().
         */
        template<typename T>
        concept Backend = alpaka::concepts::SpecializationOf<T, alpaka::Dict> && requires(T const& backend) {
            backend[object::deviceSpec];
            backend[object::exec];
        } && alpaka::concepts::SpecializationOf<internal::concepts::BackendDeviceSpec<T>, onHost::DeviceSpec> && alpaka::concepts::Executor<internal::concepts::BackendExecutor<T>>;
    } // namespace concepts

} // namespace alpaka::onHost
