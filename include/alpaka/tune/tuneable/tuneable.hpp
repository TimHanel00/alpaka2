//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/tune/utils/tupleHelper.h"

#include <alpaka/tune/concepts.hpp>

#include <string>
#include <utility>

namespace alpaka::tune
{

    namespace detail
    {
        enum class SpecialTuneableID : std::size_t
        {
            userDef = 3123,
            NumBlocks = 4321,
            ThreadBlock = 7124,
            NumFrames = 1238,
            FrameExtent = 3748,
            NoTune = 1489,
            DefaultCompileTune = 43279,
            Count
        };

        // this is runtime
        template<std::size_t N>
        std::string getNameFromTag()
        {
            static int numUserTuneables = 0;
            static int numCompileTuneables = 0;
            switch(N)
            {
            case static_cast<std::size_t>(SpecialTuneableID::NumBlocks):
                return "NumBlocksTune";
            case static_cast<std::size_t>(SpecialTuneableID::ThreadBlock):
                return "ThreadBlockTune";
            case static_cast<std::size_t>(SpecialTuneableID::NumFrames):
                return "NumFramesTune";
            case static_cast<std::size_t>(SpecialTuneableID::FrameExtent):
                return "FrameExtentTune";
            case static_cast<std::size_t>(SpecialTuneableID::DefaultCompileTune):
                return "C_Tunable " + std::to_string(numCompileTuneables++);
            default:
                break;
            }
            return "Tunable " + std::to_string(numUserTuneables++);
        }

        struct NoTune
        {
            [[nodiscard]] NoTune copy() const
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
    enum class TuneableKind
    {
        TuneableMD,
        Tuneable,
        CTuneable
    };

    template<typename T, std::size_t ID, TuneableKind kind, typename T_Storage>
    struct BaseTuneable
    {
        using ValueType = T;
        using StorageType = T_Storage;
        BaseTuneable() = default;
        virtual ~BaseTuneable() = default;
        virtual uint32_t getNumValues();
        virtual std::string getName();
        static constexpr std::size_t tag = ID;
        static constexpr auto tuneableType = kind;
        std::optional<T> startingValue;
    };

    /**
     * @brief Used to define a compile-time tuneable.
     * All tuning values are encoded as template arguments and accessible during compilation.
     * Each `CTunable` defines a fixed, immutable configuration space, represented as a `std::tuple`
     * of compile-time constants or constexpr`alpaka::Vec /alpaka::CVec` types.
     * Currently only usable in combination with a trait definition of your KernelBundle @see alpaka::tune::traits
     *
     *  Example usage:
     * @code
     * using namespace alpaka::tune;
     * #define tileSizeID=10
     * constexpr auto tileSize = CTunable<
     *     tileSizeID, // Identifier
     *     CVec<uint32_t, 20>,
     *     CVec<uint32_t, 40>,
     *     CVec<uint32_t, 80>,
     * >{};
     */

    template<std::size_t ID = static_cast<std::size_t>(detail::SpecialTuneableID::userDef), typename... T>
    struct CTunable
        : public BaseTuneable<
              typename std::tuple_element_t<0, std::tuple<T...>>::type,
              ID,
              TuneableKind::CTuneable,
              std::tuple<T...>>
    {
        using Base = BaseTuneable<
            typename std::tuple_element_t<0, std::tuple<T...>>::type,
            ID,
            TuneableKind::CTuneable,
            std::tuple<T...>>;

        static constexpr auto tag = ID;
        static constexpr auto tuneableType = TuneableKind::CTuneable;

        // All CVecs must be compatible
        static_assert(sizeof...(T) > 0, "CTunable requires at least one Parameter");
        using Tuple = std::tuple<T...>;
        using Values = Tuple; // this is the constexpr compile-time payload

        // Runtime storage (for BaseTuneable interface)
        Tuple values;
        std::string m_name = detail::getNameFromTag<ID>();

        constexpr CTunable(std::string const& name = "", Tuple initValues = Tuple{}) : values(initValues)
        {
            if(!name.empty())
                m_name = name;
        }

        // --- BaseTuneable Interface Implementations ---

        std::string getName() override
        {
            return m_name;
        }

        constexpr uint32_t getNumValues() override
        {
            return sizeof...(T);
        }
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *specialization of CTuneable to expand a tuple of values into a parameter pack -> to ease generator syntax**/

    template<std::size_t ID, typename... T>
    struct CTunable<ID, std::tuple<T...>> : CTunable<ID, T...>
    {
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *
     * Tunable` allows defining runtime-resolved tuning parameters (these can be trivially copyable objects or types)
     * it must be defined on the Host.
     * their values will be used on the device and therefore require to be trivially copyable.
     * The values are stored in a `std::vector`
     * the tuning space can be generated more easily using runtime generators (e.g., `linSpace`, `logSpace`)
     * (@see namespace alpaka::tune::generate)
     *
     * Example usage:
     * @code
     *
     * struct foo{};
     * struct bar{};
     * using namespace alpaka::tune;
     * auto tune = Tunable<ID>({foo{}, bar{}});
     * auto learningRate = Tunable({0.005, 0.02, 0.1});
     * static constexpr learningID=2;
     * auto learningRate = Tunable<learningID>(generate::logSpace(1e-6,0.1,2),0,"learning");
     * @endcode
     *
     * @tparam ID Identifier for the tuneable ->
     * must only be modified when using this tuneable in a constraint
     * ( unless using this for frameSpec tunes -> IDs for frameTuneables found under namespace: tune::frame)
     *
     *
     *
     * @tparam T Type of the values stored in the vector.
     */
    template<typename T, std::size_t ID = static_cast<std::size_t>(detail::SpecialTuneableID::userDef)>
    struct Tuneable : public BaseTuneable<T, ID, TuneableKind::TuneableMD, std::vector<T>>
    {
        using Base = BaseTuneable<T, ID, TuneableKind::TuneableMD, std::vector<T>>;
        using T_Storage = typename Base::StorageType;
        T_Storage values;
        std::optional<uint32_t> startingIndex = std::nullopt;
        std::string m_name = detail::getNameFromTag<ID>();

        std::string getName() override
        {
            return m_name;
        }

        uint32_t getNumValues() override
        {
            return values.size();
        }

        /**
         * @brief Construct from an initializer list of values.
         *
         * @param input List of values defining the tuning space for this parameter.
         * @param startingIndex Optional starting index for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr Tuneable(
            std::initializer_list<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingIndex.has_value())
            {
                this->startingIndex = startingIndex;
            }
        }

        /**
         * @brief Construct tuning space from an std::vector
         *
         * @param input List of values defining the tuning space for this parameter.
         * @param startingIndex Optional starting index for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr Tuneable(
            std::vector<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingIndex.has_value())
            {
                this->startingIndex = startingIndex;
            }
        }

        /**
         * @brief Construct from an `IdxRange<T>` to generate a tuning space.
         *
         * requires T to be either an arithmetic or alpaka::Vec type
         *
         * @param input Index range defining the start, end, and stride per dimension.
         * @param startingIndex Optional starting index.
         * @param name Optional name for the tuneable.
         */

        constexpr Tuneable(
            alpaka::IdxRange<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "") requires(tune::concepts::ArithmeticComparableOrVec<T>)
            : values(input)
        {
            for(T val = input.m_begin(); allTrue(val <= input.m_end()); val += input.m_stride)
            {
                values.push_back(input);
            }
            if(!name.empty())
                m_name = name;
            if(startingIndex.has_value())
            {
                this->startingIndex = startingIndex;
            }
        }
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *
     * does expose the same interface as @see Tuneable, yet the type is restricted to the alpaka::concepts::vector
     * the alpaka vector has to have atleast 2 dimensions
     * furthermore this tuneable allows expressing tuning dimensions using a alpaka::Vector as independent tuning
     * knobs. Meaning that all dimensions are treated as a seperate parameter in the tuning process. If this behaviour
     * is unintended, simply use a normal tuneable, as its the more generic (even if you are dealing with
     * multi-dimensional vectors) Example usage:
     * @code
     * auto tune = Tunable({Vec{8,2}, Vec{6,8}, Vec{10,12}});
     *
     * auto tune1 = Tunable(IdxRange{Vec{8,2},Vec{100,20},Vec{1,1}});
     * @endcode
     *
     * @tparam ID Identifier for the tuneable ->
     * must only be modified when using this tuneable in a constraint
     * ( unless using this for frameSpec tunes -> IDs for frameTuneables found under namespace: tune::frame)
     * @tparam T Type of the values stored in the vector.
     * */
    template<
        alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2>,
        std::size_t ID = static_cast<std::size_t>(detail::SpecialTuneableID::userDef)>
    struct TuneableMD : public BaseTuneable<T, ID, TuneableKind::TuneableMD, std::vector<T>>
    {
        using Base = BaseTuneable<T, ID, TuneableKind::TuneableMD, std::vector<T>>;
        static constexpr auto dim = alpaka::getDim(T{});
        using T_Storage = typename Base::StorageType;

        static_assert(dim > 1, "Given alpaka Vector - dimension must be higher then 1, use Tuneable instead.");

        T_Storage values;
        std::optional<uint32_t> startingIndex = std::nullopt;
        std::string m_name = detail::getNameFromTag<ID>();
        TuneableMD() = default;

        std::string getName() override
        {
            return m_name;
        }

        /// Return the number of values in the tuning space
        uint32_t getNumValues() override
        {
            return values.size();
        }

        /**
         * @brief Construct from an initializer list of alpaka::Vec.
         *
         * @param input List of alpaka::Vec defining the tuning space.
         * @param startingIndex Optional starting index for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr TuneableMD(
            std::initializer_list<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingIndex.has_value())
            {
                this->startingIndex = startingIndex;
            }
        }

        /**
         * @brief Construct from a std::vector of alpaka::Vec.
         *
         * @param input Vector of alaka::Vec defining the tuning space.
         * @param startingIndex Optional starting index for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr TuneableMD(
            std::vector<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingIndex.has_value())
            {
                this->startingIndex = startingIndex;
            }
        }

        /**
         * @brief Construct from an `IdxRange<T>` to generate a tuning space.
         *
         * Each dimension is treated independently. For example, if
         * `start = {2,3}`, `end = {4,5}`, `stride = {3,2}`, then
         * `{2,5}` is included in the generated space.
         * If this behavior is not intended, use a normal `Tuneable`.
         *
         * @param input Index range defining the start, end, and stride per dimension.
         * @param startingIndex Optional starting index.
         * @param name Optional name for the tuneable.
         */
        constexpr TuneableMD(
            IdxRange<T> input,
            std::optional<uint32_t> startingIndex = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            // this is not a exhaustive extension. only a manhatten like traverse
            T current = input.m_begin();
            rekGenerate(0, current, input);
        }

    private:
        void rekGenerate(uint32_t curDim, T& cur, IdxRange<T>& input)
        {
            if(curDim == dim)
            {
                values.push_back(cur);
                return;
            }

            auto val = cur[curDim];
            if(val > input.m_end[curDim])
                return;
            cur[curDim] = val + input.m_stride[curDim];
            rekGenerate(curDim, cur, input);
            cur[curDim] = val;
            rekGenerate(curDim + 1, cur, input);
        }
    };

    /**
     * @namespace frameTune
     * @brief Contains predefined identifier constants for special frameSpec-related tuneables.
     * They can be used to define constraints between frame related tuneables,
     * without the need to manual define identfier.
     * @code
     * concepts::TuningSession auto session=TuningBuilder{}.with....
     * session.withConstraints<frameTune::FrameExtent,
     *  frame::ThreadBlock>([&](auto frameElems,auto threads)
     * { return frameElems%threads==0; };
     *
     */
    namespace frameTune
    {
        static constexpr std::size_t numBlocks(static_cast<std::size_t>(detail::SpecialTuneableID::NumBlocks));
        static constexpr std::size_t ThreadBlock(static_cast<std::size_t>(detail::SpecialTuneableID::ThreadBlock));
        static constexpr std::size_t NumFrames(static_cast<std::size_t>(detail::SpecialTuneableID::NumFrames));
        static constexpr std::size_t FrameExtent(static_cast<std::size_t>(detail::SpecialTuneableID::FrameExtent));
    } // namespace frameTune

    // namespace alpaka::tune


} // namespace alpaka::tune

#endif // TUNEABLE_H
