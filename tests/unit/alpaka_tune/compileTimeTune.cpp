//
// Created by tim on 06.10.25.
//
//
// Created by tim on 06.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/tune/utils/compileTimeTemplates.hpp>

#include <catch2/catch_test_macros.hpp>

struct nonTrivial
{
};

template<typename CVec>
struct TestKernel
{
    using Param1 = CVec;
    TestKernel() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue() const
    {
        return CVec{};
    }
};

template<typename T, typename CVec, typename T2>
struct TestKernelThreeDim
{
    using Param1 = T;
    using Param2 = CVec;
    using Param3 = T2;
    TestKernelThreeDim() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue() const
    {
        return CVec{};
    }
};

template<typename T, typename CVec, typename CVec2>
struct TestKernelMD
{
    using Param1 = T;
    using Param2 = CVec;
    using Param3 = CVec2;
    TestKernelMD() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue_1() const
    {
        return CVec{};
    }

    auto getValue_2() const
    {
        return CVec2{};
    }
};

namespace alpaka
{


    namespace tune::trait
    {
        template<typename T>
        struct CompileTimeTuneableTrait<TestKernel<T>>
        {
            static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};
            using t = typename T::type;

            static constexpr auto tuneAbleDefinitions()
            {
                constexpr auto tune1
                    = tune::CTunable<static_cast<std::size_t>(0), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                return std::tuple{tune1};
            }
        };

        template<typename T, typename CVec_type, typename T2>
        struct CompileTimeTuneableTrait<TestKernelThreeDim<T, CVec_type, T2>>
        {
            // change Index of template parameter
            static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(1)>{};
            using t = typename CVec_type::type;

            static constexpr auto tuneAbleDefinitions()
            {
                constexpr auto tune1
                    = tune::CTunable<static_cast<std::size_t>(6), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                return std::tuple{tune1};
            }
        };

        template<typename T, typename CVec_type, typename CVec_type2>
        struct CompileTimeTuneableTrait<TestKernelMD<T, CVec_type, CVec_type2>>
        {
            // change Index of template parameter
            static constexpr auto tuned_indices
                = CVec<std::size_t, static_cast<std::size_t>(1), static_cast<std::size_t>(2)>{};
            using t = typename CVec_type::type;

            static constexpr auto tuneAbleDefinitions()
            {
                constexpr auto tune1
                    = tune::CTunable<static_cast<std::size_t>(6), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                constexpr auto tune2 = tune::CTunable<
                    static_cast<std::size_t>(12),
                    CVec<double_t, 15.1>,
                    CVec<double_t, 20.3>,
                    CVec<double_t, 200.5>>{};
                return std::tuple{tune1, tune2};
            }
        };
    } // namespace tune::trait

    TEST_CASE("parseCompileTimeTuneables", "[KernelVariantGeneration]")
    {
        TestKernel<CVec<uint32_t, 0>> tuned{};
        auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(alpaka::tune::trait::hasUserDefinedCTuneable<kernelFn>::value);
        static auto variants = typename tune::trait::RegisteredCTuneables<std::decay_t<kernelFn>>::T_KernelVariants{};
        auto vals = std::tuple{CVec<uint32_t, 1>{}, CVec<uint32_t, 2>{}, CVec<uint32_t, 8>{}, CVec<uint32_t, 10>{}};
        alpaka::tune::utils::for_each_enumerate(
            variants,
            [&]<typename T0>(T0 const& val, auto idx)
            {
                static_assert(std::is_convertible_v<typename T0::Param1, CVec<uint32_t, 18>>);
                alpaka::tune::utils::visitIndex(idx, vals, [&](auto const& val2) { CHECK(val.getValue() == val2); });
            });
    };

    TEST_CASE("parseCompileTimeTuneableChangeIndex", "[KernelVariantGenerationWithChangedIndex]")
    {
        TestKernelThreeDim<nonTrivial, CVec<uint32_t, 12>, float_t> tuned{};
        auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(alpaka::tune::trait::hasUserDefinedCTuneable<kernelFn>::value);
        static auto variants = typename tune::trait::RegisteredCTuneables<std::decay_t<kernelFn>>::T_KernelVariants{};
        auto vals = std::tuple{CVec<uint32_t, 1>{}, CVec<uint32_t, 2>{}, CVec<uint32_t, 8>{}, CVec<uint32_t, 10>{}};
        alpaka::tune::utils::for_each_enumerate(
            variants,
            [&]<typename T0>(T0 const& val, auto idx)
            {
                static_assert(std::is_same_v<typename T0::Param1, nonTrivial>);
                static_assert(std::is_convertible_v<typename T0::Param2, CVec<uint32_t, 18>>);
                static_assert(std::is_same_v<typename T0::Param3, float_t>);
                alpaka::tune::utils::visitIndex(idx, vals, [&](auto const& val2) { CHECK(val.getValue() == val2); });
            });
    };
    template<typename T_Dummy>
    struct Dummy2;

    TEST_CASE("parseCompileTimeTuneableMultiDim", "[KernelVariantGenerationWithMultipleDimensions]")
    {
        TestKernelMD<nonTrivial, CVec<uint32_t, 12>, CVec<double_t, 11.0>> tuned{};
        auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(alpaka::tune::trait::hasUserDefinedCTuneable<kernelFn>::value);
        static auto variants = typename tune::trait::RegisteredCTuneables<std::decay_t<kernelFn>>::T_KernelVariants{};
        /* manual cross product of definition:
        constexpr auto tune1
                    = tune::CTunable<static_cast<std::size_t>(6), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
        constexpr auto tune2 = tune::CTunable<
            static_cast<std::size_t>(12),
            CVec<double_t, 15.1>,
            CVec<double_t, 20.3>,
            CVec<double_t, 200.5>>{};
            */
        auto testCartesianProduct = std::tuple{
            std::tuple{CVec<uint32_t, 1>{}, CVec<double_t, 15.1>{}},
            std::tuple{CVec<uint32_t, 1>{}, CVec<double_t, 20.3>{}},
            std::tuple{CVec<uint32_t, 1>{}, CVec<double_t, 200.5>{}},

            std::tuple{CVec<uint32_t, 2>{}, CVec<double_t, 15.1>{}},
            std::tuple{CVec<uint32_t, 2>{}, CVec<double_t, 20.3>{}},
            std::tuple{CVec<uint32_t, 2>{}, CVec<double_t, 200.5>{}},

            std::tuple{CVec<uint32_t, 8>{}, CVec<double_t, 15.1>{}},
            std::tuple{CVec<uint32_t, 8>{}, CVec<double_t, 20.3>{}},
            std::tuple{CVec<uint32_t, 8>{}, CVec<double_t, 200.5>{}},

            std::tuple{CVec<uint32_t, 10>{}, CVec<double_t, 15.1>{}},
            std::tuple{CVec<uint32_t, 10>{}, CVec<double_t, 20.3>{}},
            std::tuple{CVec<uint32_t, 10>{}, CVec<double_t, 200.5>{}}};
        tune::utils::for_each(
            testCartesianProduct,
            [&](auto& tuple)
            {
                std::size_t i = tune::trait::getRtimeIndexMap(bundle)[tuple];

                alpaka::tune::runtime_Kernel_dispatch(
                    i,
                    variants,
                    [&]<typename T_KernelBundle>(T_KernelBundle&& element)
                    {
                        static_assert(std::is_same_v<typename T_KernelBundle::T, nonTrivial>);
                        static_assert(std::is_convertible_v<typename T_KernelBundle::CVec, CVec<uint32_t, 18>>);
                        static_assert(std::is_same_v<typename T_KernelBundle::CVec2, CVec<double_t, 900.0>>);
                        CHECK(std::get<0>(tuple) == element.getValue_1());
                        CHECK(std::get<1>(tuple) == element.getValue_2());
                    });
            });
    };
} // namespace alpaka
