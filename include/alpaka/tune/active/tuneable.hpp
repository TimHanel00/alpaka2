//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"

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

    template<typename T_ActiveKernel>
    void recalculateMaxRuns(T_ActiveKernel& active)
    {
        active.maxRuns = active.maxRunsDefault;

        if constexpr(T_ActiveKernel::hasNumBlocksTune())
        {
            recalculateMaxRuns_forTune(active, active.getNumBlocksTune(), "grid");
        }

        if constexpr(T_ActiveKernel::hasThreadBlockSizeTune())
        {
            recalculateMaxRuns_forTune(active, active.getThreadBlockSizeTune(), "block");
        }

        if constexpr(T_ActiveKernel::hasNumFramesTune())
        {
            recalculateMaxRuns_forTune(active, active.getNumFramesTune(), "frame");
        }

        if constexpr(T_ActiveKernel::hasFrameExtentTune())
        {
            recalculateMaxRuns_forTune(active, active.getFrameExtentTune(), "extent");
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

    void adjustToRange(auto& value, auto& begin, auto& end, auto& stride)
    {
        if constexpr(std::is_same_v<ALPAKA_TYPEOF(begin), ALPAKA_TYPEOF(value)>)
        {
            using VType = ALPAKA_TYPEOF(begin);
            auto val = value;
            if(val == begin || val == end) // special case where we can ignore the correction
                return;
            auto offset = val - begin;
            auto remainder = offset % stride;

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
    requires IsIntegral<T>
    struct FlatTuneableHandle
    {
        ;
        std::string name;
        bool userDef;
        IdxRangeHandle<T> idxRange;

        FlatTuneableHandle(T& val, std::string const& n, bool u, T& b, T& e, T& s)
            : value(val)
            , name(n)
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

    template<size_t N>
    struct StaticString
    {
        char value[N];

        constexpr StaticString(char const (&str)[N])
        {
            std::copy_n(str, N, value);
        }

        constexpr operator std::string_view() const
        {
            return std::string_view(value, N - 1); // exclude null terminator
        }
    };

//--------------------------------------
// 2. Compile-time unique name generator via macro
//--------------------------------------
#define UNIQUE_TUNEABLE_NAME(ID)                                                                                      \
    StaticString<sizeof("Tuneable_" #ID)>                                                                             \
    {                                                                                                                 \
        "Tuneable_" #ID                                                                                               \
    }

    //--------------------------------------
    // 3. Tuneable with compile-time-only name
    //--------------------------------------
    template<StaticString Name, typename T>
    struct Tuneable
    {
        T value;
        bool userDef;
        IdxRange<T, T, T> idxRange;
        static constexpr auto tag = Name;

        static constexpr std::string_view name()
        {
            return Name;
        }

        constexpr Tuneable() : value{}, userDef(false), idxRange(defaultIdxRange(T{}))
        {
        }

        constexpr Tuneable(T val) : value(val), userDef(false), idxRange(defaultIdxRange(val))
        {
        }

        constexpr Tuneable(IdxRange<T, T, T> ir) : userDef(true), idxRange(ir)
        {
            T half;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
            value = half;
            toRange();
        }

        constexpr Tuneable(T val, IdxRange<T, T, T> ir) : value(val), userDef(true), idxRange(ir)
        {
            toRange();
        }

        void toRange()
        {
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                adjustToRange(value[i], idxRange.m_begin[i], idxRange.m_end[i], idxRange.m_stride[i]);
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
            return std::string(Name) + "*" + value.toString();
        }

        bool operator==(Tuneable const& other) const
        {
            return value == other.value;
        }

        auto copy() const
        {
            return Tuneable(value, idxRange);
        }
    };

    template<StaticString Name, typename T>
    constexpr auto makeTuneable()
    {
        return Tuneable<Name, T>{};
    }

    template<StaticString Name, typename T>
    constexpr auto makeTuneable(T val)
    {
        return Tuneable<Name, T>{val};
    }

    template<StaticString Name, typename T>
    constexpr auto makeTuneable(IdxRange<T, T, T> ir)
    {
        return Tuneable<Name, T>{ir};
    }

    template<StaticString Name, typename T>
    constexpr auto makeTuneable(T val, IdxRange<T, T, T> ir)
    {
        return Tuneable<Name, T>{val, ir};
    }

    template<size_t N, typename T>
    constexpr auto makeTuneable(char const (&name)[N])
    {
        return Tuneable<StaticString<N>(name), T>{};
    }

    template<size_t N, typename T>
    constexpr auto makeTuneable(char const (&name)[N], T val)
    {
        return Tuneable<StaticString<N>(name), T>{val};
    }

    template<size_t N, typename T>
    constexpr auto makeTuneable(char const (&name)[N], IdxRange<T, T, T> ir)
    {
        return Tuneable<StaticString<N>(name), T>{ir};
    }

    template<size_t N, typename T>
    constexpr auto makeTuneable(char const (&name)[N], T val, IdxRange<T, T, T> ir)
    {
        return Tuneable<StaticString<N>(name), T>{val, ir};
    }

    // no name supplied: fallback to macro for unique names
#define makeUnnamedTuneable(Tval) makeTuneable<UNIQUE_TUNEABLE_NAME(__COUNTER__), decltype(Tval)>(Tval)

#define makeUnnamedTuneableRange(Tval, Trange)                                                                        \
    makeTuneable<UNIQUE_TUNEABLE_NAME(__COUNTER__), decltype(Tval)>(Tval, Trange)

    // Specialized tunables


    inline constexpr StaticString<9> gridSizeName{"gridSize"};
    inline constexpr StaticString<16> threadBlockSizeName{"threadBlockSize"};
    inline constexpr StaticString<10> numFramesName{"numFrames"};
    inline constexpr StaticString<12> frameExtentName{"frameExtent"};
    inline constexpr alpaka::tune::NoTune noTune{};

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumBlocksTune()
    {
        return makeTuneable<gridSizeName, T>();
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumBlocksTune(T val)
    {
        return makeTuneable<gridSizeName>(val);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumBlocksTune(IdxRange<T, T, T> ir)
    {
        return makeTuneable<gridSizeName>(ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumBlocksTune(T val, IdxRange<T, T, T> ir)
    {
        return makeTuneable<gridSizeName>(val, ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeThreadBlockSizeTune()
    {
        return makeTuneable<threadBlockSizeName, T>();
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeThreadBlockSizeTune(T val)
    {
        return makeTuneable<threadBlockSizeName>(val);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeThreadBlockSizeTune(IdxRange<T, T, T> ir)
    {
        return makeTuneable<threadBlockSizeName>(ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeThreadBlockSizeTune(T val, IdxRange<T, T, T> ir)
    {
        return makeTuneable<threadBlockSizeName>(val, ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumFramesTune()
    {
        return makeTuneable<numFramesName, T>();
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumFramesTune(T val)
    {
        return makeTuneable<numFramesName>(val);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumFramesTune(IdxRange<T, T, T> ir)
    {
        return makeTuneable<numFramesName>(ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeNumFramesTune(T val, IdxRange<T, T, T> ir)
    {
        return makeTuneable<numFramesName>(val, ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeFrameExtentTune()
    {
        return makeTuneable<frameExtentName, T>();
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeFrameExtentTune(T val)
    {
        return makeTuneable<frameExtentName>(val);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeFrameExtentTune(IdxRange<T, T, T> ir)
    {
        return makeTuneable<frameExtentName>(ir);
    }

    template<typename T = alpaka::Vec<std::size_t, 1>>
    constexpr auto makeFrameExtentTune(T val, IdxRange<T, T, T> ir)
    {
        return makeTuneable<frameExtentName>(val, ir);
    }


} // namespace alpaka::tune

#endif // TUNEABLE_H
