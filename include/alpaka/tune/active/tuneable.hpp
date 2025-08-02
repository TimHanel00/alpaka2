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
#ifdef Debug
        std::cout << label << "STEPS:  " << steps << std::endl;
        std::cout << " for tune " << tune.name() << " steps: " << steps << std::endl;
#endif
        active.maxRuns *= steps;
        if(active.maxRuns < active.maxRunsDefault)
        {
            std::cerr << " WARNING: Overflow detected during tuning space calculation, ensure "
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

    auto safeDivExZero(std::integral auto val, std::integral auto val2)
    {
        if(val2 == 0)
            return val;
        return divExZero(val, val2);
    }

    template<typename Tuneable, typename maxVec>
    void extendInputListFromPartition(
        Tuneable& tuneable,
        maxVec const& maxVal,
        maxVec const& partitionedVec,
        std::size_t minSteps = defaultMinSteps,
        std::size_t maxSteps = defaultMaxSteps)
    {
        using Vec = typename Tuneable::ValueType;
        using Scalar = typename Vec::type;

#ifdef Debug
        std::cout << "[TuneStep] minSteps = " << minSteps << ", maxSteps = " << maxSteps << "\n";
        std::cout << "[TuneStep] maxVal = " << maxVal << ", partitionedVec = " << partitionedVec << "\n";
#endif
        if(minSteps == 0 || maxSteps == 0)
        {
#ifdef Debug
            std::cout << "[EarlyExit] Skipping due to minSteps or maxSteps being 0.\n";
            return;
#endif
        }
        /*
        // Early exit: if partition is already larger than maxVal, just use maxVal
        if (!allTrue(partitionedVec < maxVal))
        {
            std::cout << "[EarlyExit] partitionedVec >= maxVal in at least one dim. Using maxVal directly.\n";
            tuneable.extendInputList({maxVal});
            return;
        }*/

        Vec baseStep = partitionedVec;
        Vec step;
#ifdef Debug
        std::cout << "[StepCalc] Starting step computation loop...\n";
#endif
        for(std::size_t i = 0; i < alpaka::getDim(step); ++i)
        {
            Scalar base = baseStep[i];
            if(base == 0)
            {
#ifdef Debug
                std::cerr << "[Warning] baseStep[" << i << "] = 0, forcing to 1.\n";
#endif
                base = 1;
            }

            Scalar rawStep = maxVal[i] / static_cast<Scalar>(maxSteps);
            double divDown = static_cast<double>(rawStep) / static_cast<double>(base);
            Scalar nDown = static_cast<Scalar>(std::floor(divDown));
            Scalar candidate = std::max(nDown * base, base);
            step[i] = candidate;
#ifdef Debug
            std::cout << "[StepCalc] Dim " << i << ":\n";
            std::cout << "  baseStep = " << baseStep[i] << ", maxVal = " << maxVal[i] << "\n";
            std::cout << "  rawStep = " << rawStep << ", divDown = " << divDown << ", nDown = " << nDown << "\n";
            std::cout << "  Initial step = " << step[i] << "\n";
#endif
            Scalar numSteps = maxVal[i] / step[i];
#ifdef Debug
            std::cout << "  numSteps = " << numSteps << " (vs minSteps = " << minSteps << ")\n";
#endif
            if(numSteps < minSteps)
            {
                Scalar minStep = maxVal[i] / static_cast<Scalar>(minSteps);
                double divUp = static_cast<double>(minStep) / static_cast<double>(base);
                Scalar nUp = static_cast<Scalar>(std::ceil(divUp));
                Scalar adjusted = nUp * base;
                step[i] = std::max(adjusted, Scalar(1));
#ifdef Debug
                std::cout << "  [Fallback] minStep = " << minStep << ", divUp = " << divUp << ", nUp = " << nUp
                          << ", adjusted = " << adjusted << ", final fallback step = " << step[i] << "\n";
#endif
                if(adjusted >= maxVal[i])
                {
                    step[i] = std::max(minStep, Scalar(1));
#ifdef Debug
                    std::cout << "  [Adjusted] Step too large. Using minStep fallback: " << step[i] << "\n";
#endif
                }
            }

            if(step[i] == 0)
            {
#ifdef Debug
                std::cerr << "[Error] Final step[" << i << "] is 0! Forcing to 1.\n";
#endif
                step[i] = 1;
            }
        }

        std::vector<Vec> values;
        Vec current = step;
        std::size_t count = 0;
#ifdef Debug
        std::cout << "[ValueGen] Generating values starting from step: " << step << "\n";
#endif
        while(allTrue(current < maxVal) && values.size() < maxSteps)
        {
#ifdef Debug
            std::cout << "  Adding value: " << current << "\n";
#endif
            values.emplace_back(current);
            current = current + step;
            ++count;
        }
#ifdef Debug
        std::cout << "NumblocksTune " << std::endl;

        std::ranges::for_each(values, [](auto const& v) { std::cout << v << ' '; });
#endif
        tuneable.extendInputList(std::move(values));
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

    inline auto absDiff(std::integral auto& a, std::integral auto& b)
    {
        return std::max(a, b) - std::min(a, b);
    }

    // manhattendistance
    template<typename TVec>
    auto l1_distance(TVec const& a, TVec const& b)
    {
        typename TVec::type sum = 0;
        for(uint32_t i = 0; i < alpaka::getDim(TVec{}); ++i)
            sum += absDiff(a[i], b[i]);
        // using std::abs causes errors on unsigned types and I want to avoid casting or taking a non std impl

        return sum;
    }

    template<typename T>
    requires(!isVector_v<T>)
    std::size_t findNearestIndex(std::vector<T> const& sortedVec, T const& target)
    {
        if(sortedVec.empty())
            throw std::runtime_error("Cannot search in an empty vector.");

        auto lower = std::lower_bound(sortedVec.begin(), sortedVec.end(), target);

        if(lower == sortedVec.begin())
            return 0;

        if(lower == sortedVec.end())
            return sortedVec.size() - 1;

        std::size_t idx = std::distance(sortedVec.begin(), lower);
        T const& high = *lower;
        T const& low = *(lower - 1);
        return absDiff(high, target) < absDiff(low, target) ? idx : idx - 1;
    }

    template<typename TVec>
    requires isVector_v<TVec>
    std::size_t findNearestIndex(std::vector<TVec> const& vecList, TVec const& target)
    {
        if(vecList.empty())
            throw std::runtime_error("Cannot search in an empty vector.");

        std::size_t bestIndex = 0;
        auto bestDist = l1_distance(vecList[0], target);

        for(std::size_t i = 1; i < vecList.size(); ++i)
        {
            auto dist = l1_distance(vecList[i], target);
            if(dist < bestDist)
            {
                bestDist = dist;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

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
        uint32_t index;

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
            this->index = findNearestIndex(valList.get(), val);
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
        uint32_t index;

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
            this->index = findNearestIndex(valList.get(), val);
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

    template<std::size_t Tag = 0, typename... CVecs>
    struct CTunable
    {
        // Sanity checks
        static constexpr auto tag = Tag;
        static_assert(sizeof...(CVecs) > 0, "CTunable requires at least one CVec");

        // Check all are CVec
        static_assert((alpaka::isCVector_v<CVecs> && ...), "All parameters to CTunable must be CVec<T, ...>");

        // Extract scalar type from first CVec
        using Scalar = typename std::tuple_element_t<0, std::tuple<CVecs...>>::type;
        static constexpr std::size_t dim = std::tuple_element_t<0, std::tuple<CVecs...>>::dim();
        // Ensure all CVecs use the same scalar type
        static_assert(
            (std::is_same_v<Scalar, typename CVecs::type> && ...),
            "All CVecs in CTunable must have the same scalar type");

        // Store all CVecs
        using Tuple = std::tuple<CVecs...>;
        using Values = Tuple;
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
        using ValueType = T; // alpaka vector per default
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

        void extendInputList(std::vector<T>&& baseInputList)
        {
            inputList.reserve(inputList.size() + baseInputList.size());
            std::move(baseInputList.begin(), baseInputList.end(), std::back_inserter(inputList));
        }

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

        constexpr Tuneable(
            std::vector<T> const& input,
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

        [[nodiscard]] std::string toString() const
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
        bool v_inList = false;
        for(T const& v : inputList)
        {
            v_inList = (v == value) ? true : v_inList;
#ifdef Debug
            std::cout << " inputList for " << this->name() << v.toString() << std::endl;
#endif
            dependentList.push_back(v);
        }
        if(!v_inList)
        {
            dependentList.push_back(value);
        }
        dependentList.erase(std::unique(dependentList.begin(), dependentList.end()), dependentList.end());
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
        bool v_inList = false;
        for(T const& v : inputList)
        {
            v_inList = (v == value) ? true : v_inList;
#ifdef Debug
            std::cout << " inputList for " << this->name() << v.toString() << std::endl;
#endif
            for(std::size_t dim = 0; dim < D; ++dim)
            {
                auto val = v[dim];
                independentLists[dim].push_back(val);
            }
        }
        if(!v_inList)
        {
            for(std::size_t dim = 0; dim < D; ++dim)
            {
                auto val = value[dim];
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
