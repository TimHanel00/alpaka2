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

        template<typename T>
        concept Hashable = requires(T const& t) {
            { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
        };

        /**
         * @brief Computes a byte-level hash of any trivially copyable value ( this must be ensured by the caller)
         *
         * Implements the 64-bit FNV-1a hash algorithm for hashing raw object bytes.
         * Fast, constexpr-friendly, and suitable for deterministic non-cryptographic hashing.
         *
         * @see https://datatracker.ietf.org/doc/html/draft-eastlake-fnv
         *
         * @tparam T  Type of the value to hash (should be trivially copyable).
         * @param[in] value  Object whose bytes will be hashed.
         * @return 64-bit FNV-1a hash of the object's memory representation.
         */
        template<typename T>
        constexpr std::size_t hashBytes(T const& value)
        {
            auto const* ptr = reinterpret_cast<unsigned char const*>(&value);
            std::size_t hash = 1'469'598'103'934'665'603ULL;
            for(std::size_t i = 0; i < sizeof(T); ++i)
                hash = (hash ^ ptr[i]) * 1'099'511'628'211ULL;
            return hash;
        }

        template<typename T>
        struct Hasher
        {
            static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable!");
            constexpr Hasher() = default;

            std::size_t operator()(T const& val) const
            {
                if constexpr(Hashable<T>)
                {
                    return std::hash<T>{}(val);
                }
                else
                {
                    return hashBytes(val);
                }
            }
        };

        /**
         * @brief Combines a hash value into an existing hash seed.
         *
         * Implements the Boost `hash_combine` pattern to mix the hash of a value
         * into an accumulated hash seed.
         *
         * @see https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
         *
         * @tparam T  Type of the value to hash.
         * @param[in,out] seed  Current hash seed to be updated.
         * @param[in] value Value to incorporate into the hash.
         */
        template<class T>
        constexpr void hash_combine(std::size_t& seed, T const& value)
        {
            Hasher<T> hasher;
            seed ^= hasher(value) + 0x9e37'79b9 + (seed << 6) + (seed >> 2);
        }

        // overload to perform boost hash combine on a std::vector given a seed
        template<typename T>
        constexpr void hash_combine(std::size_t& seed, std::vector<T> const& values)
        {
            for(auto const& value : values)
            {
                hash_combine(seed, value);
            }
        }

        template<typename T>
        constexpr std::size_t hashType(std::size_t& seed)
        {
            for(auto const ch : onHost::demangledName<T>())
            {
                detail::hash_combine(seed, ch);
            }
            return seed;
        }

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
