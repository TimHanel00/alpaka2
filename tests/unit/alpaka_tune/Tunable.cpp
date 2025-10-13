//
// Created by tim on 10.10.25.
//
#include "alpaka/tune/tuneable/Tunable.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <tuple>

using namespace alpaka::tune;

///
///
/// ** CTuneable Tests ** ///
///
///
TEST_CASE("CTuneable basic compile-time tuple", "[CTuneable]")
{
    using t = uint32_t;
    using Tune = CTunable<10, alpaka::CVec<t, 0>, alpaka::CVec<t, 1>, alpaka::CVec<t, 2>>;

    // Compile-time tuple type
    static_assert(Tune::dim == 1u);
    static_assert(Tune::tag == 10u);

    // Compile-time number of values
    constexpr auto numVals = std::tuple_size_v<Tune::Values>;
    static_assert(numVals == 3);

    // Compile-time getValueByIndex
    constexpr auto v0 = Tune::getValueByIndex<0>();
    constexpr auto v1 = Tune::getValueByIndex<1>();
    constexpr auto v2 = Tune::getValueByIndex<2>();
    static_assert(std::is_same_v<decltype(v0), alpaka::CVec<t, 0> const>);
    static_assert(std::is_same_v<decltype(v1), alpaka::CVec<t, 1> const>);
    static_assert(std::is_same_v<decltype(v2), alpaka::CVec<t, 2> const>);
}

TEST_CASE("CTuneable runtime interface", "[CTuneable]")
{
    using t = double_t;
    CTunable<140, alpaka::CVec<t, 0.1>, alpaka::CVec<t, 1.1>, alpaka::CVec<t, 2.4>> tuneDefault;
    std::cout << tuneDefault.getName() << std::endl;
    CHECK(tuneDefault.getName() == "C_Tunable 0"); // default name
    auto numVals = tuneDefault.getNumValues();
    CHECK(numVals[0] == 3); // tuple size

    CTunable<12, alpaka::CVec<t, 0.1>, alpaka::CVec<t, 1.1>, alpaka::CVec<t, 2.4>> tuneNamed("MyTune");
    CHECK(tuneNamed.getName() == "MyTune");
}

TEST_CASE("CTuneable supports user-defined types", "[CTuneable]")
{
    struct Foo
    {
        int a{};

        constexpr bool operator==(Foo const&) const
        {
            return true;
        }
    };

    struct Bar
    {
        double b{};

        constexpr bool operator==(Bar const&) const
        {
            return true;
        }
    };

    using Tune = CTunable<20, Foo, Bar>;
    static_assert(std::is_same_v<typename Tune::Values, std::tuple<Foo, Bar>>);

    CTunable<20, Foo, Bar> tune("CustomTune");
    CHECK(tune.getName() == "CustomTune");
    auto numVals = tune.getNumValues();
    CHECK(numVals[0] == 2); // tuple size
}

///
///
/// ** TunableMD Tests ** ///
///
///
using Vec2u = alpaka::Vec<uint32_t, 2u>;
using Vec3u = alpaka::Vec<uint32_t, 3u>;

TEST_CASE("TunableMD - construction from initializer list", "[TunableMD]")
{
    TunableMD<Vec2u> tmd{{{1u, 10u}, {2u, 20u}, {3u, 15u}}, std::optional<Vec2u>{{2u, 20u}}, "InitListTuneable"};

    CHECK(tmd.getName() == "InitListTuneable");

    auto numVals = tmd.getNumValues();
    CHECK(numVals[0] == 3u);
    CHECK(numVals[1] == 3u);

    // check that each dimension is sorted
    CHECK(std::is_sorted(tmd.values[0].begin(), tmd.values[0].end()));
    CHECK(std::is_sorted(tmd.values[1].begin(), tmd.values[1].end()));

    // ensure correct lookup
    Vec2u idx{1u, 2u};
    auto val = tmd.getValueByIndex(idx);
    CHECK(val[0] == tmd.values[0][1]);
    CHECK(val[1] == tmd.values[1][2]);
}

TEST_CASE("TunableMD - construction from vector of Vec", "[TunableMD]")
{
    std::vector<Vec2u> v = {{1u, 10u}, {4u, 40u}, {2u, 20u}};
    TunableMD<Vec2u> tmd(v, std::nullopt, "VectorTuneable");

    CHECK(tmd.getName() == "VectorTuneable");
    auto numVals = tmd.getNumValues();
    CHECK(numVals[0] == 3u);
    CHECK(numVals[1] == 3u);

    // should be sorted by generateSortedSpaces
    CHECK(std::is_sorted(tmd.values[0].begin(), tmd.values[0].end()));
    CHECK(std::is_sorted(tmd.values[1].begin(), tmd.values[1].end()));
}

TEST_CASE("TunableMD - construction from IdxRange", "[TunableMD]")
{
    Vec3u begin{1, 10, 100};
    Vec3u end{3, 14, 104};
    Vec3u stride{1, 2, 2};
    alpaka::IdxRange<Vec3u> range(begin, end, stride);

    TunableMD<Vec3u> tmd(range, std::nullopt, "RangeTuneable");

    CHECK(tmd.getName() == "RangeTuneable");

    // each dimension filled correctly
    CHECK(tmd.values[0] == std::vector<uint32_t>({1, 2, 3}));
    CHECK(tmd.values[1] == std::vector<uint32_t>({10, 12, 14}));
    CHECK(tmd.values[2] == std::vector<uint32_t>({100, 102, 104}));
}

TEST_CASE("TunableMD - findStartingIndex valid value", "[TunableMD]")
{
    Vec2u start{2u, 20u};
    TunableMD<Vec2u> tmd{{{1u, 10u}, {2u, 20u}, {3u, 30u}}, start, ""};

    CHECK(tmd.startingIndex.has_value());
    auto idx = tmd.startingIndex.value();
    CHECK(idx[0u] == 1u);
    CHECK(idx[1u] == 1u);
}

TEST_CASE("TunableMD - findStartingIndex invalid value", "[TunableMD]")
{
    Vec2u start{999u, 999u};
    TunableMD<Vec2u> tmd{{{1u, 10u}, {2u, 20u}, {3u, 30u}}, start};

    CHECK(!tmd.startingIndex.has_value());
}

TEST_CASE("TunableMD - 3D consistency check", "[TunableMD]")
{
    std::vector<Vec3u> space = {{1u, 10u, 100u}, {2u, 20u, 200u}, {3u, 30u, 300u}};

    TunableMD<Vec3u> tmd(space, std::nullopt, "3DTest");

    CHECK(tmd.getNumValues()[0] == 3u);
    CHECK(tmd.getNumValues()[1] == 3u);
    CHECK(tmd.getNumValues()[2] == 3u);

    Vec3u idx{2, 1, 0};
    auto val = tmd.getValueByIndex(idx);

    CHECK(val[0] == 3u);
    CHECK(val[1] == 20u);
    CHECK(val[2] == 100u);
}

///
///
/// ** Tunable Tests ** ///
///
///
TEST_CASE("Tunable basic construction from initializer list", "[Tunable]")
{
    auto t = Tunable<int, 1001u>{{1, 2, 3, 4, 5}};
    CHECK(t.getNumValues()[0] == 5);
    CHECK(t.getValueByIndex({0u}) == 1);
    CHECK(t.getValueByIndex({4u}) == 5);
    CHECK_FALSE(t.startingIndex.has_value());
}

TEST_CASE("Tunable with explicit name and starting value", "[Tunable]")
{
    Tunable<int, 2001> t({10, 20, 30, 40}, 30, "MyIntTuneable");
    CHECK(t.getName() == "MyIntTuneable");
    CHECK(t.getNumValues()[0] == 4);
    CHECK(t.startingIndex.has_value());
    CHECK(t.startingIndex.value() == 2u);
}

TEST_CASE("Tunable constructed from vector", "[Tunable]")
{
    std::vector<int> vals = {3, 6, 9};
    Tunable<int, 3001> t(vals, 6, "VectorTuneable");
    CHECK(t.getName() == "VectorTuneable");
    CHECK(t.getNumValues()[0] == 3);
    CHECK(t.startingIndex == std::optional<uint32_t>{1u});
    CHECK(t.getValueByIndex({2u}) == 9);
}

TEST_CASE("Tunable constructed from IdxRange", "[Tunable]")
{
    using alpaka::IdxRange;
    using Vec1 = alpaka::Vec<uint32_t, 1>;
    auto range = IdxRange{Vec1{1u}, Vec1{5u}, Vec1{1u}}; // generates [1,2,3,4,5]
    auto tune = Tunable<Vec1>(range, Vec1{3u}, "RangeTuneable");
    CHECK(tune.getName() == "RangeTuneable");
    CHECK(tune.getNumValues()[0] == 5);
    CHECK(tune.startingIndex == std::optional<uint32_t>{2u});
    CHECK(tune.getValueByIndex(Vec1{4u}) == Vec1{5});
}

TEST_CASE("Tunable handles missing starting value gracefully", "[Tunable]")
{
    Tunable<int, 5001> t({1, 2, 3}, 999, "InvalidStart");
    CHECK_FALSE(t.startingIndex.has_value());
}

TEST_CASE("Tunable with Vec type", "[Tunable][Vec]")
{
    using Vec2 = alpaka::Vec<unsigned int, 2>;
    std::vector<Vec2> values = {Vec2{1u, 2u}, Vec2{3u, 4u}, Vec2{5u, 6u}};
    Tunable<Vec2, 6001> t(values, Vec2{3u, 4u}, "VecTuneable");

    CHECK(t.getName() == "VecTuneable");
    CHECK(t.getNumValues()[0] == 3);
    CHECK(t.startingIndex == std::optional<uint32_t>{1u});
    CHECK(t.getValueByIndex({1u})[0u] == 3u);
    CHECK(t.getValueByIndex({1u})[1u] == 4u);
}
