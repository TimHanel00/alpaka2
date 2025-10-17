//
// Created by tim on 17.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/tune/tuneable/Tunable.hpp>
#include <alpaka/tune/tuneable/frameSpecTuningModel.hpp>

#include <catch2/catch_test_macros.hpp>
using namespace alpaka;
using namespace alpaka::tune;
namespace frame = alpaka::tune::frame;

TEST_CASE("default construction disables all tuneables", "[FrameSpecTuningModel][create]")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{1, 1}, Vec<uint32_t, 2u>{2, 2}};
    FrameSpecTuningModel model{spec};

    REQUIRE_FALSE(model.hasNumFramesTune());
    REQUIRE_FALSE(model.hasFrameExtentTune());
    REQUIRE_FALSE(model.hasNumBlocksTune());
    REQUIRE_FALSE(model.hasNumThreadsTune());
}

TEST_CASE("builder enables tuneable and get*Tune returns stored object", "[FrameSpecTuningModel][getters][builders]")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};

    // prepare tuneables
    TunableMD<tune::frame::numFrames, V2u> numFramesTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<tune::frame::frameExtent, V2u> frameExtentTune{{V2u{4, 4}, V2u{8, 8}}};
    TunableMD<tune::frame::numBlocks, V2u> numBlocksTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<tune::frame::numThreads, V2u> numThreadsTune{{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune(numFramesTune)
                     .withFrameExtentTune(frameExtentTune)
                     .withNumBlocksTune(numBlocksTune)
                     .withNumThreadsTune(numThreadsTune);

    // has* checks
    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // get* checks (returned tuneables should match what was provided)
    auto nf = tuned.getNumFramesTune();
    auto fe = tuned.getFrameExtentTune();
    auto nb = tuned.getNumBlocksTune();
    auto nt = tuned.getNumThreadsTune();

    REQUIRE(nf.getName() == numFramesTune.getName());
    REQUIRE(fe.getName() == frameExtentTune.getName());
    REQUIRE(nb.getName() == numBlocksTune.getName());
    REQUIRE(nt.getName() == numThreadsTune.getName());

    REQUIRE(nf.getNumValues()[0u] == numFramesTune.getNumValues()[0u]);
    REQUIRE(fe.getNumValues()[0u] == frameExtentTune.getNumValues()[0u]);
}

TEST_CASE(
    "get*Tune returns dummy when default-enabled with no explicit tuneable",
    "[FrameSpecTuningModel][getters][default]")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{1, 1}, Vec<uint32_t, 2u>{2, 2}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune()
                     .withFrameExtentTune()
                     .withNumBlocksTune()
                     .withNumThreadsTune();

    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // getTune calls should compile and return a tuneable-like object
    auto nf = tuned.getNumFramesTune();
    auto fe = tuned.getFrameExtentTune();
    auto nb = tuned.getNumBlocksTune();
    auto nt = tuned.getNumThreadsTune();

    // minimal runtime checks — we just assert these are distinct types (compile-time check in templates)
    STATIC_REQUIRE(!std::is_same_v<decltype(nf), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(fe), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(nb), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(nt), bool>);
}

TEST_CASE("partial enabling only allows get for enabled tuneables", "[FrameSpecTuningModel][getters][partial]")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{2, 2}, Vec<uint32_t, 2u>{4, 4}};

    auto partial = FrameSpecTuningModel{spec}.withNumThreadsTune();

    REQUIRE_FALSE(partial.hasNumFramesTune());
    REQUIRE_FALSE(partial.hasFrameExtentTune());
    REQUIRE_FALSE(partial.hasNumBlocksTune());
    REQUIRE(partial.hasNumThreadsTune());

    // Only getNumThreadsTune should compile; others should trigger static_assert if used.
    auto nt = partial.getNumThreadsTune();
    STATIC_REQUIRE(!std::is_same_v<decltype(nt), bool>);
}

TEST_CASE("rvalue-qualified builder chaining and getter correctness", "[FrameSpecTuningModel][getters][rvalue]")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};

    FrameSpecTuningModel model{spec};

    auto tuned = std::move(model).withNumFramesTune().withFrameExtentTune();

    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE_FALSE(tuned.hasNumBlocksTune());
    REQUIRE_FALSE(tuned.hasNumThreadsTune());

    // should compile and return tuneable-like objects
    auto nf = tuned.getNumFramesTune();
    auto fe = tuned.getFrameExtentTune();
    STATIC_REQUIRE(!std::is_same_v<decltype(nf), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(fe), bool>);
}

TEST_CASE("end-to-end: custom tuneables are preserved through chaining", "[FrameSpecTuningModel][integration]")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{16, 16}};

    TunableMD<tune::frame::frameExtent, V2u> frameExtentTune{{V2u{8, 8}, V2u{16, 16}}};
    TunableMD<tune::frame::numThreads, V2u> numThreadsTune{{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}.withFrameExtentTune(frameExtentTune).withNumThreadsTune(numThreadsTune);

    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumThreadsTune());
    REQUIRE_FALSE(tuned.hasNumFramesTune());
    REQUIRE_FALSE(tuned.hasNumBlocksTune());

    auto fe = tuned.getFrameExtentTune();
    auto nt = tuned.getNumThreadsTune();

    REQUIRE(fe.getName() == frameExtentTune.getName());
    REQUIRE(nt.getName() == numThreadsTune.getName());
    REQUIRE(fe.getNumValues()[0u] == frameExtentTune.getNumValues()[0u]);
    REQUIRE(nt.getNumValues()[0u] == numThreadsTune.getNumValues()[0u]);
}

TEST_CASE("builder chaining: all tuneables retrieved correctly", "[FrameSpecTuningModel][getters][chain]")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};

    TunableMD<tune::frame::numFrames, V2u> numFramesTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<tune::frame::frameExtent, V2u> frameExtentTune{{V2u{4, 4}, V2u{8, 8}}};
    TunableMD<tune::frame::numBlocks, V2u> numBlocksTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<tune::frame::numThreads, V2u> numThreadsTune{{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune(numFramesTune)
                     .withFrameExtentTune(frameExtentTune)
                     .withNumBlocksTune(numBlocksTune)
                     .withNumThreadsTune(numThreadsTune);

    // has* must all be true
    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // get* must return tuneables with expected names and sizes
    REQUIRE(tuned.getNumFramesTune().getName() == numFramesTune.getName());
    REQUIRE(tuned.getFrameExtentTune().getName() == frameExtentTune.getName());
    REQUIRE(tuned.getNumBlocksTune().getName() == numBlocksTune.getName());
    REQUIRE(tuned.getNumThreadsTune().getName() == numThreadsTune.getName());
}
