//
// Created by tim on 08.10.25.
//
#include "alpaka/tune/tunable/Tunable.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
using namespace alpaka::tune;

TEST_CASE("Runtime LinSpace generates correct sequence", "[LinSpace]")
{
    auto lin = generate::linSpace(1, 5, 1);
    std::vector expected{1, 2, 3, 4, 5};
    REQUIRE(lin == expected);
}

TEST_CASE("Runtime LogSpace generates correct sequence", "[LogSpace]")
{
    auto log = generate::logSpace(1, 16, 2);
    std::vector expected{1, 2, 4, 8, 16};
    REQUIRE(log == expected);
}

TEST_CASE("Runtime LinSpace on Vector")
{
    using vecT = alpaka::Vec<uint32_t, 2u>;
    auto log = generate::linSpace(vecT{1, 5}, vecT{6, 7}, vecT{2, 2});
    std::vector expected{vecT{1, 5}, vecT{3, 7}};
    REQUIRE(log == expected);
}
template<typename T>
struct Dummy;

TEST_CASE("Compile-time linear space generates correct tuple", "[c_LinSpace]")
{
    using type = uint32_t;
    using seq_T = generate::c_LinSpace<type, 2, 5>::values;
    using ExpectedTuple
        = std::tuple<alpaka::Vec<type, 2>, alpaka::Vec<type, 3>, alpaka::Vec<type, 4>, alpaka::Vec<type, 5>>;
    static_assert(std::is_same_v<seq_T, ExpectedTuple>);
}

TEST_CASE("Compile-time logarithmic space generates correct tuple", "[c_LogSpace]")
{
    using type = uint32_t;
    constexpr auto seq = generate::c_LogSpace<type, 1, 9, 3>::values{};
    using ExpectedTuple = std::tuple<alpaka::Vec<type, 1>, alpaka::Vec<type, 3>, alpaka::Vec<type, 9>> const;
    using tuple = decltype(seq);
    static_assert(std::is_same_v<tuple, ExpectedTuple>);
}

TEST_CASE("Tuneable compatible", "[c_LogSpace]")
{
    using type = int;
    using TuneTuple = CTunable<static_cast<std::size_t>(0), generate::c_LogSpace<type, 1, 9, 3>::values>::Values;
    using ExpectedTuple = std::tuple<alpaka::Vec<int, 1>, alpaka::Vec<int, 3>, alpaka::Vec<int, 9>>;
    utils::for_each_enumerate(
        TuneTuple{},
        []<std::size_t I>(auto&& elem)
        {
            using SeqType = std::remove_cvref_t<decltype(elem)>;
            using ExpectedType = std::tuple_element_t<I, ExpectedTuple>;
            CHECK(SeqType{} == ExpectedType{});
        });
}
