//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"

#include <alpaka/tune/utils/VecUtils.h>
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
        std::cout << " for tune " << tune.name() << " steps: " << steps << std::endl;
        active.maxRuns *= steps;
        if(active.maxRuns < active.maxRunsDefault)
        {
            std::cout << " WARNING: Overflow detected during tuning space calculation, ensure "
                         "you have a max NumofRuns selected!"
                      << std::endl;
            active.maxRuns = UINT64_MAX;
        }
    }

    template <typename ValueVec, typename IdxRange>
void clampToSpec_elem(const ValueVec& maxVal, IdxRange& idxRange, bool isThreadBlockTune, auto& device)
{
    using Scalar = typename ValueVec::type;
    constexpr auto dim = ValueVec::dim();
    using dimType=decltype(dim);
    const Scalar warpSize = device.getDeviceProperties().m_warpSize;
    const Scalar mpCount = device.getDeviceProperties().m_multiProcessorCount;
    const Scalar maxThreads = device.getDeviceProperties().m_maxThreadsPerBlock;

    for (auto i = static_cast<dimType>(0); i < dim; ++i)
    {
        auto& begin = idxRange.m_begin[i];
        auto& stride = idxRange.m_stride[i];
        auto& end = idxRange.m_end[i];
        const auto& max = maxVal[i];

        bool adjusted = false;

        // Begin or stride too large
        if (begin >= max || stride >= max)
        {
            adjusted = true;
            if (isThreadBlockTune)
            {
                begin = primeFactorPartitioning(warpSize, Scalar{});
            }
            else
            {
                begin = primeFactorPartitioning(mpCount, Scalar{});
            }
            stride = begin;

            // If still invalid, fallback
            if (begin >= max)
            {
                begin = 1;
                stride = 1;
                end = max;
                continue;
            }
        }

        // End is too high or invalid
        if (end > max || end < begin)
        {
            adjusted = true;
            Scalar b = (max - begin) / stride;
            end = begin + b * stride;

            if (end < begin)
            {
                end = begin;
            }
        }

        // Clamp end for thread block case
        if (isThreadBlockTune && end > maxThreads)
        {
            adjusted = true;
            end = std::min(end, maxThreads);
        }

        // Final fallback if clamping failed
        if (adjusted && (begin >= max || stride >= max))
        {
            begin = 1;
            stride = 1;
            end = max;
        }
    }
}
    template<typename T_NumFrames,typename T_NumThreads,typename T_activeKernel>
    void clampToSpec(auto & device,onHost::FrameSpec<T_NumFrames,T_NumThreads>& frameSpec, T_activeKernel& activeKernel)
    {
        if constexpr(T_activeKernel::hasNumBlocksTune())
        {
            auto countMps=device.getDeviceProperties().m_multiProcessorCount;
            clampToSpec_elem(frameSpec.m_numFrames, activeKernel.getNumBlocksTune().idxRange, false, device);
            activeKernel.getNumBlocksTune().toRange();
        }
        if constexpr(T_activeKernel::hasThreadBlockSizeTune())
        {
            auto countWarps=device.getDeviceProperties().m_warpSize;
            device.getDeviceProperties().m_maxThreadsPerBlock;
            clampToSpec_elem(frameSpec.m_frameExtent, activeKernel.getThreadBlockSizeTune().idxRange, true, device);

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
    concept isIntegral = std::is_integral_v<T>;
    template<typename T>
    concept isFloating = std::is_floating_point_v<T>;

    template<typename T, uint32_t Dim>
    struct RefStorage
    {
        std::array<std::reference_wrapper<T>, Dim> refs;
        RefStorage() = delete;

        constexpr RefStorage(alpaka::Vec<T, Dim>& vec) : refs{makeRefs(vec, std::make_index_sequence<Dim>{})}
        {
        }

        template<std::size_t... Is>
        static constexpr std::array<std::reference_wrapper<T>, Dim> makeRefs(
            alpaka::Vec<T, Dim>& vec,
            std::index_sequence<Is...>)
        {
            return {std::ref(vec[Is])...};
        }

        constexpr RefStorage(isIntegral auto& value) : refs{std::ref(value)}
        {
        }

        constexpr RefStorage(isFloating auto& value) : refs{std::ref(value)}
        {
        }

        constexpr T& operator[](std::size_t i)
        {
            return refs[i].get();
        }

        constexpr T const& operator[](std::size_t i) const
        {
            return refs[i].get();
        }
    };

    template<typename T>
    concept IsIntegral = std::is_integral_v<T>;

    template<typename T_Vec>
    struct IdxRangeHandle
    {
        using T_Storage = typename T_Vec::Storage;
        T_Vec m_begin;
        T_Vec m_end;
        T_Vec m_stride;

        IdxRangeHandle(auto& b, auto& e, auto& s) : m_begin(T_Storage(b)), m_end(T_Storage(e)), m_stride(T_Storage(s))
        {
        }
    };
    template<typename T, bool = alpaka::isVector_v<T>>
    struct TuneableHandle;

    template<typename T>
    struct TuneableHandle<T, true>
    {
        std::string name;
        bool userDef;
        using T_Storage = RefStorage<typename T::type, alpaka::getDim(T{})>;
        using T_Vec = Vec<typename T::type, alpaka::getDim(T{}), T_Storage>;
        static constexpr auto dim = ::alpaka::getDim(T{});
        IdxRangeHandle<T_Vec> idxRange;

        TuneableHandle(T& val, std::string n, bool u, T& b, T& e, T& s)
            : value(T_Storage(val))
            , name(std::move(n))
            , userDef(u)
            , idxRange(b, e, s)
        {
        }

        T_Vec value;
    };

    template<typename T>
    struct TuneableHandle<T, false>
    {
        std::string name;
        bool userDef;
        using T_Storage = RefStorage<T, 1>;
        using T_Vec = Vec<T, 1, T_Storage>;
        static constexpr auto dim = 1;
        IdxRangeHandle<T_Vec> idxRange;

        TuneableHandle(T& val, std::string n, bool u, T& b, T& e, T& s)
            : value(T_Vec(T_Storage(val)))
            , name(std::move(n))
            , userDef(u)
            , idxRange(b, e, s)
        {
        }

        T_Vec value;
    };

    /**
     *this is a 1 dim non-owning tuple handle for a tuneable object - these are used for defining strategies and
     *allowing uniform access treating every dimension of any tunable as uniform
     * @tparam T a primitive type used to store the reference to the tuneable object in a tuple
     *
     */
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
    inline constexpr std::size_t getId()
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

    template<auto N>
    struct wrapper
    {
    };

    template<int N>
    struct Tag;

    template<int N>
    Tag<N> register_tag(Tag<N>);

    template<typename T, typename = void>
    struct is_registered : std::false_type
    {
    };

    template<typename T>
    struct is_registered<T, decltype(register_tag(std::declval<T>()), void())> : std::true_type
    {
    };

    template<
        typename Begin,
        typename End,
        typename Stride,
        std::size_t ID = static_cast<std::size_t>(SpecialTuneableID::userDef)>
    struct CTunable
    {
        // Check they are all CVec
        static_assert(alpaka::isCVector_v<Begin>, "CTuneable construction failed: Begin must be a CVec");
        static_assert(alpaka::isCVector_v<End>, "CTuneable construction failed:  End must be a CVec");
        static_assert(alpaka::isCVector_v<Stride>, "CTuneable construction failed:  Stride must be a CVec");

        // Get dimensions
        static constexpr std::size_t dimBegin = std::tuple_size_v<typename Begin::Storage::Values>;
        static constexpr std::size_t dimEnd = std::tuple_size_v<typename End::Storage::Values>;
        static constexpr std::size_t dimStride = std::tuple_size_v<typename Stride::Storage::Values>;

        static_assert(
            dimBegin == dimEnd && dimEnd == dimStride,
            "CTuneable construction failed:  Begin, End, and Stride must have the same number of dimensions");

        // Get scalar types
        using ScalarBegin = typename Begin::type;
        using ScalarEnd = typename End::type;
        using ScalarStride = typename Stride::type;

        static_assert(
            std::is_same_v<ScalarBegin, ScalarEnd> && std::is_same_v<ScalarEnd, ScalarStride>
                && std::is_same_v<ScalarStride, ScalarBegin>,
            "CTuneable construction failed:  types of Begin, End, and Stride must match");

        // This is valid now
        using T_Begin = Begin;
        using T_End = End;
        using T_Stride = Stride;
        static constexpr std::size_t tag = ID;
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
        static constexpr std::size_t tag = ID;
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
        constexpr Tuneable(T numSteps, T start, T end, std::string const& name = "")
            : value(end)
            , userDef(true)
            , idxRange(start, end, T::all(1))
        {
            if(!name.empty())
                m_name = name;

            IdxRange<T, T, T> stepsRange(start, end, T::all(1));
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                if(numSteps[i] > 1)
                    stepsRange.m_stride[i] = (stepsRange.m_end[i] - stepsRange.m_begin[i]) / (numSteps[i] - 1);
                else
                    stepsRange.m_stride[i] = 0;
                stepsRange.m_end[i] = stepsRange.m_begin[i] + stepsRange.m_stride[i] * (numSteps[i] - 1);
            }
            idxRange = stepsRange;
            toRange();
        }

        // numSteps -> explicit init
        constexpr Tuneable(T numSteps, T start, T end, T init, std::string const& name = "")
            : value(init)
            , userDef(true)
            , idxRange(start, end, T::all(1))
        {
            if(!name.empty())
                m_name = name;

            IdxRange<T, T, T> stepsRange(start, end, T::all(1));
            T steps{};
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                if(numSteps[i] > 1)
                    stepsRange.m_stride[i] = (stepsRange.m_end[i] - stepsRange.m_begin[i]) / (numSteps[i] - 1);
                else
                    stepsRange.m_stride[i] = 0;
                stepsRange.m_end[i] = stepsRange.m_begin[i] + stepsRange.m_stride[i] * (numSteps[i] - 1);
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
