//
// Created by tim on 17.10.25.
//

#ifndef TUNEABLEHELPER_H
#define TUNEABLEHELPER_H
#include "alpaka/UniqueId.hpp"

#include <string>

namespace alpaka::tune
{
    namespace detail
    {
        enum class SpecialTuneableID : std::size_t
        {
            userDef = alpaka::uniqueId(),
            numBlocks = alpaka::uniqueId(),
            numThreads = alpaka::uniqueId(),
            numFrames = alpaka::uniqueId(),
            frameExtent = alpaka::uniqueId(),
            NoTune = alpaka::uniqueId(),
            DefaultCompileTune = alpaka::uniqueId(),
            Count
        };
    } // namespace detail

    namespace detail
    {


        template<std::size_t N>
        std::string getNameFromTag_comp()
        {
            static int numCompileTuneables = 0;
            if(N == static_cast<std::size_t>(SpecialTuneableID::DefaultCompileTune))
            {
                return "C_Tunable " + std::to_string(numCompileTuneables++);
            }
            return "C_Tunable " + std::to_string(N);
        }

        // this is runtime
        template<std::size_t N>
        std::string getNameFromTag()
        {
            static int numUserTuneables = 0;

            switch(N)
            {
            case static_cast<std::size_t>(SpecialTuneableID::userDef):
                return "Tunable " + std::to_string(numUserTuneables++);
            case static_cast<std::size_t>(SpecialTuneableID::numBlocks):
                return "NumBlocksTune";
            case static_cast<std::size_t>(SpecialTuneableID::numThreads):
                return "ThreadBlockTune";
            case static_cast<std::size_t>(SpecialTuneableID::numFrames):
                return "NumFramesTune";
            case static_cast<std::size_t>(SpecialTuneableID::frameExtent):
                return "FrameExtentTune";
            default:
                break;
            }
            return "Tunable " + std::to_string(N);
        }

        struct NoTune
        {
            [[nodiscard]] static NoTune copy()
            {
                return NoTune{};
            }

            [[nodiscard]] static std::string toHash()
            {
                return "";
            }
        };

        template<typename T>
        constexpr bool is_NoTune_v = std::is_same_v<T, NoTune>;

        inline constexpr alpaka::tune::detail::NoTune noTune{};
    } // namespace detail

    template<typename TuneableA, typename TuneableB>
    constexpr bool isSameTuneable(TuneableA const& a, TuneableB const& b)
    {
        return TuneableA::tag == TuneableB::tag;
    }

    /**
     * @brief kind of tuneable to ease compile-time handling**/
    enum class TunableKind
    {
        TunableMD,
        Tunable,
        CTunable,
        Dummy
    };

    namespace detail
    {
        /*
         * basically holds and stores an ID -> used to "enable" frameTuneables without specific typing, while
         */
        template<auto ID>
        struct ShallowTunableDummy
        {
            ShallowTunableDummy() = default;
            static constexpr auto tag = ID;
            static constexpr auto dim = 1;
            static constexpr auto tuneableType = TunableKind::Dummy;

            auto getNumValues() const
            {
                return Vec<uint32_t, 1u>{1u};
            }
        };
    } // namespace detail
} // namespace alpaka::tune
#endif // TUNEABLEHELPER_H
