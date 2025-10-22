//
// Created by tim on 08.10.25.
//

#ifndef CONCEPTS_H
#define CONCEPTS_H

// #include <alpaka/tune/IO/runTimeHistory.hpp>
#include <alpaka/tune/interfaces/MetricInterface.hpp>
#include <alpaka/tune/tunable/Tunable.hpp>

namespace alpaka::tune::concepts
{
    template<typename T>
    concept Floating = std::is_floating_point_v<T>;
    template<typename T>
    concept Integral = std::is_integral_v<T>;
    template<typename T>
    concept ArithmeticComparable = requires(T a, T b) {
        { a + b } -> std::same_as<T>;
        { a < b } -> std::convertible_to<bool>;
        { a <= b } -> std::convertible_to<bool>;
        { a == b } -> std::convertible_to<bool>;
    };
    template<typename T>
    concept ArithmeticComparableOrVec = ArithmeticComparable<T> || isVector_v<T>;
} // namespace alpaka::tune::concepts

// forward declare Config
namespace alpaka::tune::config
{
    template<alpaka::tune::concepts::Integral T, uint32_t NumTunables>
    struct Config;
    template<alpaka::tune::concepts::Floating T, uint32_t NumTunables>
    struct NormalizedConfig;
} // namespace alpaka::tune::config

namespace alpaka::tune::concepts
{
    template<typename T>
    concept shallowTunable = std::is_same_v<alpaka::tune::detail::ShallowTunableDummy<T::tag>, T>;
    template<typename T>
    concept runtimeTuneable =
        // must have a static member `tuneableType`
        requires {
            { T::tuneableType } -> std::convertible_to<TunableKind>;
        }
        // must be a runtime tuneable, i.e., kind == Tuneable or TunableMD
        && (T::tuneableType == TunableKind::Tunable || T::tuneableType == TunableKind::TunableMD);
    // currently compile-time tunables are not permitted
    template<typename T>
    concept TuneableLike = runtimeTuneable<T> || std::is_same_v<T, alpaka::tune::detail::ShallowTunableDummy<T::tag>>
                           || std::is_same_v<T, detail::NoTune>;
    template<typename T>
    concept MetricInterface = requires(T t) {
        { t.returnComparison } -> std::convertible_to<detail::returnComparison>;
        { t.start() } -> std::same_as<void>;
        { t.end() } -> std::same_as<double_t>;
        //{ t.end(std::declval<R&>(), std::declval<S&>()) } -> std::same_as<void>;
    }; // namespace concepts


    template<typename T>
    concept ConfigLike = []<typename U = std::remove_cvref_t<T>>
    {
        if constexpr(requires {
                         typename U::value_type;
                         { U::size() } -> std::convertible_to<std::size_t>;
                     })
        {
            using V = typename U::value_type;
            constexpr auto N = U::size();

            if constexpr(alpaka::tune::concepts::Floating<V>)
            {
                return std::is_same_v<U, alpaka::tune::config::NormalizedConfig<V, N>>;
            }
            else
            {
                // Only allow integer configs if V is integral
                return std::is_same_v<U, alpaka::tune::config::Config<V, N>>;
            }
        }
        else
        {
            return false;
        }
    }();


    template<typename T>
    concept KernelTuningModel = requires(T t) {
        // static member
        { T::numDims } -> std::convertible_to<std::size_t>;

        // instance members
        { t.m_numValues } -> std::convertible_to<std::array<uint32_t, T::numDims>>;

        // functions
        // Note: we don’t constrain argument types precisely here, since they’re templated
        { t.getValuesFromConfig(std::declval<alpaka::tune::config::Config<uint32_t, T::numDims>>()) };
        { t.createConfigFromNormalized(std::declval<alpaka::tune::config::NormalizedConfig<double, T::numDims>>()) };
    };

    template<class Tuple, class = void>
    struct isValidFrameTuple : std::false_type
    {
    };

    template<class Tuple>
    struct isValidFrameTuple<Tuple,
        std::void_t<decltype(
            []<typename... Ts>(std::tuple<Ts...>*)
            {
                using namespace alpaka::tune::concepts;

                // 1️⃣  Reject compile-time tuneables
                static_assert(((Ts::tuneableType != TunableKind::CTunable) && ...),
                    "no compile time tuneables are currently allowed as frame tuneables!");

                // 2️⃣  Pairwise checks among all runtime tuneables
                ([]<typename A, typename... Rest>()
                {
                    // For each A in Ts...
                    ([]<typename B>()
                    {
                        if constexpr (runtimeTuneable<A> && runtimeTuneable<B>)
                        {
                            static_assert(A::dim == B::dim,
                                "All frame tuneables must have identical ::dim");

                            static_assert(
                                std::is_convertible_v<typename A::value_type, typename B::value_type> ||
                                std::is_convertible_v<typename B::value_type, typename A::value_type>,
                                "All frame tuneables must have mutually convertible ::value_type");
                        }
                    }.template operator()<Rest>(), ...);
                }.template operator()<Ts, Ts...>(), ...);
            }((Tuple*)nullptr)
        )>>
        : std::true_type
    {
    };

    template<class Tuple>
    inline constexpr bool isValidFrameTuple_v = isValidFrameTuple<Tuple>::value;


} // namespace alpaka::tune::concepts
#endif // CONCEPTS_H
