//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"

#include <alpaka/tune/utils/partitioning.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>

template<typename Tuple, typename F, std::size_t... I>
void for_each_impl(Tuple&& tup, F&& f, std::index_sequence<I...>)
{
    (f(std::get<I>(std::forward<Tuple>(tup))), ...);
}

template<typename Tuple, typename F>
void for_each(Tuple&& tup, F&& f)
{
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_impl(std::forward<Tuple>(tup), std::forward<F>(f), std::make_index_sequence<N>{});
}

#define DEFINE_TUNE_NAME(name)                                                                                        \
    inline constexpr ::alpaka::tune::StaticString<sizeof(#name)> name##_ss()                                          \
    {                                                                                                                 \
        #name                                                                                                         \
    }

#define cStr(literal)                                                                                                 \
    StaticString<sizeof(literal)>                                                                                     \
    {                                                                                                                 \
        literal                                                                                                       \
    }
//--------------------------------------
// 2. Compile-time unique name generator via macro
//--------------------------------------
#define UNIQUE_TUNEABLE_NAME(ID)                                                                                      \
    StaticString<sizeof("Tuneable_" #ID)>                                                                             \
    {                                                                                                                 \
        "Tuneable_" #ID                                                                                               \
    }

namespace alpaka::tune
{
    template<typename T_Tune, typename T_ActiveKernel>
    void recalculateMaxRuns_forTune(T_ActiveKernel& active, T_Tune const& tune, char const* label)
    {
        auto const steps = tune.numSteps();
        std::cout << label << "STEPS:  " << steps << std::endl;

        active.maxRuns *= steps;
        if(active.maxRuns < active.maxRunsDefault)
        {
            std::cout << " WARNING: Overflow detected during tuning space calculation, ensure "
                         "you have a max NumofRuns selected!"
                      << std::endl;
            active.maxRuns = UINT64_MAX;
        }
    }

    void clampToSpec_elem(auto& value, auto& idxRange)
    {
        using rangeType = ALPAKA_TYPEOF(idxRange.m_begin);
        for(std::size_t i = 0; i < alpaka::getDim(rangeType{}); ++i)
        {
            if(idxRange.m_begin[i] <= 0 || idxRange.m_begin[i] >= value[i])
            {
                idxRange.m_begin[i] = 1;
            }

            if(idxRange.m_stride[i] <= 0)
            {
                idxRange.m_stride[i] = 1;
            }
            if(idxRange.m_end[i] > value[i] || idxRange.m_end[i] < idxRange.m_begin[i])
            {
                auto maxEnd = ((value[i] - idxRange.m_begin[i]) / idxRange.m_stride[i]) * idxRange.m_stride[i]
                              + idxRange.m_begin[i];
                idxRange.m_end[i] = (maxEnd > idxRange.m_begin[i]) ? maxEnd : idxRange.m_begin[i];
            }
        }
    }

    template<typename T_activeKernel>
    void clampToSpec(auto& frameSpec, T_activeKernel& activeKernel)
    {
        if constexpr(T_activeKernel::hasNumBlocksTune())
        {
            clampToSpec_elem(frameSpec.m_numFrames, activeKernel.getNumBlocksTune().idxRange);
            activeKernel.getNumBlocksTune().toRange();
        }
        if constexpr(T_activeKernel::hasThreadBlockSizeTune())
        {
            clampToSpec_elem(frameSpec.m_frameExtent, activeKernel.getThreadBlockSizeTune().idxRange);
            activeKernel.getThreadBlockSizeTune().toRange();
        }
    }

    template<typename T>
    constexpr T mod(T a, T b)
    {
        if constexpr(std::is_floating_point_v<T>)
            return std::fmod(a, b);
        else
            return a % b;
    }

    void adjustToRange(auto& value, auto& begin, auto& end, auto& stride)
    {
        if constexpr(std::is_same_v<ALPAKA_TYPEOF(begin), ALPAKA_TYPEOF(value)>)
        {
            using VType = ALPAKA_TYPEOF(begin);
            auto val = value;
            if(val == begin || val == end) // special case where we can ignore the correction
                return;
            auto offset = val - begin;
            auto remainder = mod(offset, stride);

            if(remainder == VType{0} && val >= begin && val <= end)
                return;
            auto n = offset / stride;
            if(remainder * VType{2} >= stride)
            {
                n = n + VType{1}; // Round up if closer
            }

            auto corrected = begin + n * stride;

            // Clamp within range
            if(corrected < begin)
                corrected = begin;
            if(corrected > end)
                corrected = end;
            value = corrected;
        }

        else
        {
            throw std::runtime_error(
                std::string("Types dont match:  ") + typeid(value).name() + " vs. " + typeid(begin).name()
                + std::to_string(value) + " begin " + begin.toString());
        }
    }

    struct StorageTuneable
    {
        std::string name;
        std::string value;

        [[nodiscard]] std::string toHash() const
        {
            return name + "*" + value;
        }
    };

    template<typename T>
    concept IsIntegral = std::is_integral_v<T>;

    template<typename T>
    struct IdxRangeHandle
    {
        T& m_begin;
        T& m_end;
        T& m_stride;

        IdxRangeHandle(T& begin, T& end, T& stride) : m_begin(begin), m_end(end), m_stride(stride)
        {
        }
    };

    /**
     *this is a 1 dim non-owning tuple handle for a tuneable object - these are used for defining strategies and
     *allowing uniform access treating every dimension of any tunable as uniform
     * @tparam T a primitive type used to store the reference to the tuneable object in a tuple
     *
     */
    template<typename T>
    struct FlatTuneableHandle
    {
        std::string name;
        bool userDef;
        IdxRangeHandle<T> idxRange;

        FlatTuneableHandle(T& val, std::string n, bool u, T& b, T& e, T& s)
            : value(val)
            , name(std::move(n))
            , userDef(u)
            , idxRange(b, e, s)
        {
        }

        T& value;
    };

    inline std::size_t globalId = 0;

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

    enum class SpecialTuneableID : std::size_t
    {
        userDef = 0,
        NumBlocks = 1,
        ThreadBlock = 2,
        NumFrames = 3,
        FrameExtent = 4,
        NoTune = 5,
        Count
    };

    namespace frameTune
    {
        static constexpr std::size_t numBlocks(static_cast<std::size_t>(SpecialTuneableID::NumBlocks));
        static constexpr std::size_t ThreadBlock(static_cast<std::size_t>(SpecialTuneableID::ThreadBlock));
        static constexpr std::size_t NumFrames(static_cast<std::size_t>(SpecialTuneableID::NumFrames));
        static constexpr std::size_t FrameExtent(static_cast<std::size_t>(SpecialTuneableID::FrameExtent));
    } // namespace frameTune

    template<SpecialTuneableID T>
    constexpr std::size_t toI()
    {
        return static_cast<std::size_t>(T);
    }

    template<std::size_t N>
    constexpr std::size_t getId()
    {
        if constexpr(N == 0)
        {
            constexpr std::size_t tag = __COUNTER__;
            return tag;
        }
        else
        {
            return N;
        }
    }

    inline bool operator==(std::size_t lhs, SpecialTuneableID rhs)
    {
        return lhs == static_cast<std::size_t>(rhs);
    }

    // this is runtime
    template<std::size_t N>
    std::string getNameFromTag()
    {
        static int numTuneables = 0;
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
        default:
            break;
        }
        return "Tunable " + std::to_string(numTuneables++);
    }

    struct DimensionTraversePolicy
    {
    };

    struct DimensionsIndependent : public DimensionTraversePolicy
    {
        static constexpr bool dimensionIndependent = false;
    };

    struct DimensionsDependent : public DimensionTraversePolicy
    {
        static constexpr bool dimensionIndependent = true;
    };

    template<typename TuneableA, typename TuneableB>
    constexpr bool isSameTuneable(TuneableA const& a, TuneableB const& b)
    {
        return TuneableA::tag == TuneableB::tag;
    }

    template<
        typename Value,
        typename Begin,
        typename End,
        typename Stride,
        std::size_t ID = static_cast<std::size_t>(SpecialTuneableID::userDef),
        typename dimensionTraversePolicy = DimensionsIndependent>
    struct CTunable
    {
        using value = Value;
        using idxRange = IdxRange<Begin, End, Stride>;
        static constexpr std::size_t tag = getId<ID>();
    };

    //--------------------------------------
    // 3. Tuneable
    //--------------------------------------
    template<
        typename T = alpaka::Vec<std::size_t, 1>,
        std::size_t ID = static_cast<std::size_t>(SpecialTuneableID::userDef),
        typename dimensionTraversePolicy = DimensionsIndependent>
    struct Tuneable
    {
        using ValueType = T;
        using dimensionTraversePolicy_type = dimensionTraversePolicy;
        T value;
        bool userDef;
        static constexpr std::size_t tag = getId<ID>();
        IdxRange<T, T, T> idxRange;
        std::string m_name = getNameFromTag<ID>();
        DimensionTraversePolicy policy;
        template<typename TuneableA, typename TuneableB>
        friend constexpr bool isSameTuneable(TuneableA const& a, TuneableB const& b);

        std::string name() const
        {
            return m_name;
        }

        std::string name()
        {
            return m_name;
        }

        void toRange()
        {
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                adjustToRange(value[i], idxRange.m_begin[i], idxRange.m_end[i], idxRange.m_stride[i]);
        }

        constexpr Tuneable() : value{}, userDef(false), idxRange(defaultIdxRange(T{}))
        {
        }

        constexpr Tuneable(T init, std::string const& name = "")
            : value(init)
            , userDef(true)
            , idxRange(defaultIdxRange(init))
        {
            if(!name.empty())
                m_name = name;
            toRange();
        }

        // Value + optional range + optional name
        constexpr Tuneable(T init, IdxRange<T, T, T> ir, std::string const& name = "")
            : value(init)
            , userDef(true)
            , idxRange(ir)
        {
            if(!name.empty())
                m_name = name;
            toRange();
        }

        // Only range + optional name
        constexpr Tuneable(IdxRange<T, T, T> ir, std::string const& name = "")
            : value(ir.m_end)
            , userDef(true)
            , idxRange(ir)
        {
            if(!name.empty())
                m_name = name;
            toRange();
        }

        // From integral steps (delegating to the "steps" constructor)
        constexpr Tuneable(std::size_t integralSteps, T start, T end, T init, std::string const& name = "")
            : Tuneable(primeFactorPartitioning(integralSteps, T{}), start, end, init, name)
        {
        }

        constexpr Tuneable(std::size_t integralSteps, T start, T end, std::string const& name = "")
            : Tuneable(primeFactorPartitioning(integralSteps, T{}), start, end, end, name)
        {
        }

        // provide steps in a vector
        constexpr Tuneable(T numSteps, T start, T end, std::string const& name = "") : value(end), userDef(true)
        {
            if(!name.empty())
                m_name = name;

            IdxRange<T, T, T> stepsRange(start, end, T{});
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                stepsRange[i].m_stride = (end[i] - start[i]) / numSteps[i];
            }
            idxRange = stepsRange;
            toRange();
        }

        // numSteps -> explicit init
        constexpr Tuneable(T numSteps, T start, T end, T init, std::string const& name = "")
            : value(init)
            , userDef(true)
        {
            if(!name.empty())
                m_name = name;

            IdxRange<T, T, T> stepsRange(start, end, T{});
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                stepsRange[i].m_stride = (end[i] - start[i]) / numSteps[i];
            }
            idxRange = stepsRange;
            toRange();
        }

        static constexpr IdxRange<T, T, T> defaultIdxRange(T const& val)
        {
            T zero{}, one{};
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                zero[i] = 0;
                one[i] = 1;
            }
            return IdxRange<T, T, T>{zero, val, one};
        }

        [[nodiscard]] std::size_t numSteps() const
        {
            std::size_t numSteps = 1;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                numSteps *= ((idxRange.m_end[i] - idxRange.m_begin[i]) / idxRange.m_stride[i]) + 1;
            }
            return numSteps;
        }

        [[nodiscard]] std::string toHash() const
        {
            return m_name + "*" + value.toString();
        }

        template<typename T_Tuneable>
        bool operator==(T_Tuneable const& other) const
        {
            return (tag == T_Tuneable::tag) && (value == other.value);
        }

        auto copy() const
        {
            return Tuneable<ValueType, tag, dimensionTraversePolicy_type>(value, idxRange);
        }
    };

    // no name supplied: fallback to macro for unique names
    // Specialized tunables
    inline constexpr alpaka::tune::NoTune noTune{};


} // namespace alpaka::tune

namespace alpaka::concepts
{
    template<typename T>
    concept tuneable = requires(T t) {
        typename T::ValueType;
        typename T::dimensionTraversePolicy_type;
        T::tag;
        t.value;
        t.idxRange;
    };
} // namespace alpaka::concepts
#endif // TUNEABLE_H
