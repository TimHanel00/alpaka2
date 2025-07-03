//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/tune/utils/tupleHelper.h"

#include <alpaka/tune/utils/VecUtils.h>
#include <alpaka/tune/utils/partitioning.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>


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

    template<typename Vec>
    bool allTrue(Vec const& v)
    {
        for(std::size_t i = 0; i < Vec::dim(); ++i)
        {
            if(!v[i])
                return false;
        }
        return true;
    }

    // Scalar bool overload
    inline bool allTrue(bool const& v)
    {
        return v;
    }

#define defaultMinSteps 16
#define defaultMaxSteps 32

    auto calcNumStepsFromRange(auto const& tuneable)
    {
        return ((tuneable.idxRange.m_end - tuneable.idxRange.m_begin) / tuneable.idxRange.m_stride).product() + 1;
    }

    template<typename Tuneable, typename maxVec, typename ScalarPartitioning>
    void adaptRangeToNumSteps(
        Tuneable& tuneable, // the tuneable we want to partition
        maxVec const& maxVal, // maximum value (ndim vector)
        ScalarPartitioning partition, // the scalar ressource that has to be partitioned for m_begin and m_stride
        std::size_t minSteps = defaultMinSteps,
        std::size_t maxSteps = defaultMaxSteps)
    {
        using Vec = decltype(tuneable.idxRange.m_begin);
        using Scalar = typename Vec::type;

        Vec base = primeFactorPartitioning(partition, Vec{});

        tuneable.idxRange.m_begin = base;
        tuneable.idxRange.m_stride = base;
        tuneable.idxRange.m_end = multipleOfPartitioning(maxVal, base);

        tuneable.toRange(); // calculate numSteps

        int steps = calcNumStepsFromRange(tuneable);

        // Try scaling base up until steps are in range or we overshoot
        Scalar scale = 1;
        Vec current = base;

        while(steps > maxSteps && allTrue(current * Scalar{2} < maxVal))
        {
            current = current * Scalar{2};
            tuneable.idxRange.m_begin = current;
            tuneable.idxRange.m_stride = current;
            tuneable.idxRange.m_end = multipleOfPartitioning(maxVal, current);
            tuneable.toRange();
            steps = calcNumStepsFromRange(tuneable);
        }

        // Back off if we overshot
        while(steps < minSteps && scale > 1)
        {
            current = current / Scalar{2};
            tuneable.idxRange.m_begin = current;
            tuneable.idxRange.m_stride = current;
            tuneable.idxRange.m_end = multipleOfPartitioning(maxVal, current);
            tuneable.toRange();
            steps = calcNumStepsFromRange(tuneable);
        }
        if(!allTrue(tuneable.idxRange.m_end > tuneable.idxRange.m_begin))
        {
            // fallback to the smallest valid range
            tuneable.idxRange.m_end = tuneable.idxRange.m_begin;
        }
    }

    template<typename ValueVec, typename IdxRange>
    void clampToSpec_elem(ValueVec const& maxVal, IdxRange& idxRange, bool isThreadBlockTune, auto& device)
    {
        using Scalar = typename ValueVec::type;
        constexpr auto dim = ValueVec::dim();
        using dimType = decltype(dim);
        Scalar const warpSize = device.getDeviceProperties().m_warpSize;
        Scalar const mpCount = device.getDeviceProperties().m_multiProcessorCount;
        Scalar const maxThreads = device.getDeviceProperties().m_maxThreadsPerBlock;
        for(auto i = static_cast<dimType>(0); i < dim; ++i)
        {
            auto& begin = idxRange.m_begin[i];
            auto& stride = idxRange.m_stride[i];
            auto& end = idxRange.m_end[i];
            auto const& max = maxVal[i];

            bool adjusted = false;

            // Begin or stride too large
            if(begin > max || stride > max)
            {
                adjusted = true;
                if(isThreadBlockTune)
                {
                    idxRange.m_begin = primeFactorPartitioning(warpSize, ValueVec{});
                }
                else
                {
                    idxRange.m_stride = primeFactorPartitioning(mpCount, ValueVec{});
                }
                stride = begin;

                // If still invalid, fallback
                if(begin >= max)
                {
                    begin = 1;
                    stride = 1;
                    end = max;
                    continue;
                }
            }

            // End is too high or invalid
            if(end > max || end < begin)
            {
                adjusted = true;
                Scalar b = (max - begin) / stride;
                end = begin + b * stride;

                if(end < begin)
                {
                    end = begin;
                }
            }

            // Clamp end for thread block case
            if(isThreadBlockTune && end > maxThreads)
            {
                adjusted = true;
                end = std::min(end, maxThreads);
            }

            // Final fallback if clamping failed
            if(adjusted && (begin >= max || stride >= max))
            {
                begin = 1;
                stride = 1;
                end = max;
            }
        }
    }

    template<typename T_NumFrames, typename T_NumThreads, typename T_activeKernel>
    void clampToSpec(
        auto& device,
        onHost::FrameSpec<T_NumFrames, T_NumThreads>& frameSpec,
        T_activeKernel& activeKernel)
    {
        if constexpr(T_activeKernel::hasNumFramesTune())
        {
            clampToSpec_elem(frameSpec.m_numFrames, activeKernel.getNumFramesTune().idxRange, false, device);
            if(activeKernel.getNumFramesTune().hasRange)
                activeKernel.getNumFramesTune().toRange();
        }
        if constexpr(T_activeKernel::hasFrameExtentTune())
        {
            clampToSpec_elem(frameSpec.m_frameExtent, activeKernel.getFrameExtentTune().idxRange, false, device);
            if(activeKernel.getFrameExtentTune().hasRange)
                activeKernel.getFrameExtentTune().toRange();
        }
        if constexpr(T_activeKernel::hasNumBlocksTune())
        {
            auto countMps = device.getDeviceProperties().m_multiProcessorCount;
            clampToSpec_elem(
                frameSpec.m_threadSpec.m_numBlocks,
                activeKernel.getNumBlocksTune().idxRange,
                false,
                device);
            if(activeKernel.getNumBlocksTune().hasRange)
                activeKernel.getNumBlocksTune().toRange();
        }
        if constexpr(T_activeKernel::hasThreadBlockSizeTune())
        {
            auto countWarps = device.getDeviceProperties().m_warpSize;
            clampToSpec_elem(
                frameSpec.m_threadSpec.m_numThreads,
                activeKernel.getThreadBlockSizeTune().idxRange,
                true,
                device);

            if(activeKernel.getThreadBlockSizeTune().hasRange)
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
        static constexpr auto dim = alpaka::getDim(T{});

        IdxRangeHandle<T_Vec> idxRange;
        std::reference_wrapper<std::vector<T>> valList;
        T_Vec value;

        auto& getValues()
        {
            return valList.get();
        }

        TuneableHandle(T& val, std::string n, bool u, std::reference_wrapper<std::vector<T>> vec, T& b, T& e, T& s)
            : name(std::move(n))
            , userDef(u)
            , idxRange(b, e, s)
            , valList(vec)
            , value(T_Storage(val)) // moved last for safe order
        {
        }
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
        std::reference_wrapper<std::vector<T>> valList;
        T_Vec value;

        auto& getValues()
        {
            return valList.get();
        }

        TuneableHandle(T& val, std::string n, bool u, std::reference_wrapper<std::vector<T>> vec, T& b, T& e, T& s)
            : name(std::move(n))
            , userDef(u)
            , idxRange(b, e, s)
            , valList(vec)
            , value(T_Vec(T_Storage(val))) // moved to the end
        {
        }
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
    template<typename T, typename Policy>
    struct ValueListType;

    template<typename T>
    struct ValueListType<T, DimensionsDependent>
    {
        using type = std::vector<T>;
    };

    template<typename T>
    struct ValueListType<T, DimensionsIndependent>
    {
        static constexpr std::size_t Dim = alpaka::getDim(T{});
        using type = std::array<std::vector<typename T::type>, Dim>;
    };

    template<typename VecRef>
    struct tuneableListWrapper
    {
        std::size_t id; // index in the original tuneables tuple
        std::size_t dim; // 0 for dependent, or actual dimension for independent
        VecRef list; // reference to the std::vector<T>
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
        bool userDef = true;
        bool hasRange = true;
        static constexpr std::size_t tag = ID;
        IdxRange<T, T, T> idxRange;
        using index_type = typename T::index_type;
        static constexpr auto vecDim = alpaka::getDim(T{});
        std::vector<T> inputList;
        template<
            typename U = dimensionTraversePolicy,
            std::enable_if_t<std::is_same_v<U, DimensionsDependent>, int> = 0>
        auto makeList() -> std::vector<T>;
        template<
            typename U = dimensionTraversePolicy,
            std::enable_if_t<std::is_same_v<U, DimensionsIndependent>, int> = 0>
        auto makeList() -> std::array<std::vector<typename T::type>, vecDim>;
        std::string m_name = getNameFromTag<ID>();
        DimensionTraversePolicy policy;
        template<typename TuneableA, typename TuneableB>
        friend constexpr bool isSameTuneable(TuneableA const& a, TuneableB const& b);
        using ValueList = typename ValueListType<T, dimensionTraversePolicy>::type;
        ValueList valueList;

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

        constexpr Tuneable(
            std::initializer_list<T> input,
            std::optional<T> init = std::nullopt,
            std::string const& name = "")
            : inputList(input)
            , idxRange{T::all(1), T::all(1), T::all(1)}
            , hasRange(false)
        {
            if(!name.empty())
                m_name = name;
            if(init.has_value())
            {
                value = init.value();
            }
            else
            {
                value = inputList[0];
            }
        }

        constexpr Tuneable(IdxRange<T, T, T> range, std::optional<T> init = std::nullopt, std::string const& name = "")
            : idxRange(range)
        {
            if(!name.empty())
                m_name = name;
            if(init.has_value())
                value = init.value();
            else
            {
                value = range.m_end;
            }
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

        auto expand(std::size_t tuneableId)
        {
            if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsDependent>)
            {
                return std::tuple<tuneableListWrapper<std::vector<T>&>>{{tuneableId, 0, valueList}};
            }
            else if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsIndependent>)
            {
                using ValueType = typename T::type;

                return [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    return std::make_tuple(
                        tuneableListWrapper<std::vector<ValueType>&>{tuneableId, Is, valueList[Is]}...);
                }(std::make_index_sequence<vecDim>{});
            }
        }

        bool removeIfValid(std::size_t index, std::size_t dim)
        {
            if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsDependent>)
            {
                if(index >= valueList.size())
                    throw std::out_of_range("Index out of bounds in removeIfValid (Dependent)");

                if(valueList[index] == value)
                    return false;

                valueList.erase(valueList.begin() + index);
                return true;
            }
            else if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsIndependent>)
            {
                if(dim >= vecDim)
                    throw std::out_of_range("Dimension out of bounds in removeIfValid (Independent)");

                auto& vec = valueList[dim];

                if(index >= vec.size())
                    throw std::out_of_range("Index out of bounds in removeIfValid (Independent)");

                if(vec[index] == value[dim])
                    return false;

                vec.erase(vec.begin() + index);
                return true;
            }
        }

        [[nodiscard]] std::size_t numSteps() const
        {
            if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsDependent>)
            {
                return valueList.size(); // valueList is std::vector<T>
            }
            else if constexpr(std::is_same_v<dimensionTraversePolicy, DimensionsIndependent>)
            {
                std::size_t total = 1;
                for(std::size_t i = 0; i < vecDim; ++i)
                {
                    total *= valueList[i].size(); // valueList is std::array<std::vector<...>, vecDim>
                }
                return total;
            }
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

    template<typename T, std::size_t ID, typename dimensionTraversePolicy>
    template<typename U, std::enable_if_t<std::is_same_v<U, DimensionsDependent>, int>>
    auto Tuneable<T, ID, dimensionTraversePolicy>::makeList() -> std::vector<T>
    {
        std::vector<T> dependentList;
        if(hasRange)
        {
            for(auto i = idxRange.m_begin; allTrue(i <= idxRange.m_end); i += idxRange.m_stride)
            {
                dependentList.emplace_back(i);
            }
        }
        for(T const& v : inputList)
        {
            dependentList.push_back(v);
        }
        valueList = dependentList;
        return dependentList;
    }

    template<typename T, std::size_t ID, typename dimensionTraversePolicy>
    template<typename U, std::enable_if_t<std::is_same_v<U, DimensionsIndependent>, int>>
    auto Tuneable<T, ID, dimensionTraversePolicy>::makeList()
        -> std::array<std::vector<typename T::type>, Tuneable<T, ID, dimensionTraversePolicy>::vecDim>
    {
        using Scalar = typename T::type;
        constexpr std::size_t D = vecDim;
        std::array<std::vector<Scalar>, D> independentLists;

        if(hasRange)
        {
            for(std::size_t dim = 0; dim < D; ++dim)
            {
                auto begin = idxRange.m_begin[dim];
                auto end = idxRange.m_end[dim];
                auto stride = idxRange.m_stride[dim];

                if(stride == 0)
                {
                    independentLists[dim].push_back(begin);
                }

                for(auto val = begin; val <= end; val += stride)
                {
                    independentLists[dim].push_back(val);
                }
            }
        }

        for(T const& v : inputList)
        {
            for(std::size_t dim = 0; dim < D; ++dim)
            {
                auto val = v[dim];
                independentLists[dim].push_back(val);
            }
        }


        for(std::size_t dim = 0; dim < D; ++dim)
        {
            auto& list = independentLists[dim];
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }

        valueList = independentLists;
        return independentLists;
    }


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
