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
        if constexpr(activeKernel.hasNumBlocksTune())
        {
            clampToSpec_elem(frameSpec.m_numFrames, activeKernel.getNumBlocksTune().idxRange);
            activeKernel.getNumBlocksTune().toRange();
        }
        if constexpr(activeKernel.hasThreadBlockSizeTune())
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
     *allowing uniform access
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

    template<typename T = alpaka::Vec<std::size_t, 1>>
    struct Tuneable
    {
        T value;
        std::string name;
        bool userDef;
        IdxRange<T, T, T> idxRange;

        static IdxRange<T, T, T> defaultIdxRange(T const& val)
        {
            T zero{}, one{};
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                zero[i] = 0;
                one[i] = 1;
            }
            return IdxRange<T, T, T>{zero, val, one};
        }

        Tuneable() : value{}, name("TuneableVec: default"), userDef(false), idxRange(defaultIdxRange(T{}))
        {
        }

        explicit Tuneable(T const& val)
            : value(val)
            , name("TuneableVec: ")
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {
        }

        Tuneable(T const& val, std::string iname)
            : value(val)
            , name(std::move(iname))
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {
        }

        Tuneable(T const& val, IdxRange<T, T, T> ir) : value(val), name("TuneableVec: "), userDef(true), idxRange(ir)
        {
            toRange();
        }

        Tuneable(T const& val, std::string iname, IdxRange<T, T, T> ir)
            : value(val)
            , name(std::move(iname))
            , userDef(true)
            , idxRange(ir)
        {
            toRange();
        }

        Tuneable(std::string iname, IdxRange<T, T, T> ir)
            : name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            T half;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
            value = half;
            toRange();
        }

        Tuneable(IdxRange<T, T, T> ir) : name("TuneableVec: "), userDef(true), idxRange(std::move(ir))
        {
            T half;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
            value = half;
            toRange();
        }

        void toRange()
        {
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
                adjustToRange(value[i], idxRange.m_begin[i], idxRange.m_end[i], idxRange.m_stride[i]);
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
            return name + "*" + value.toString();
        }

        bool operator==(Tuneable const& other) const
        {
            return value == other.value && name == other.name;
        }

        auto copy()
        {
            return Tuneable(value, name, idxRange);
        }
    };

    // Specialized tunables


    template<typename T = alpaka::Vec<std::size_t, 1>>
    struct NumBlocksTune : public Tuneable<T>
    {
        explicit NumBlocksTune() : Tuneable<T>(T(64), "gridSize")
        {
        }

        explicit NumBlocksTune(T initial_value) : Tuneable<T>(initial_value, "gridSize")
        {
        }

        explicit NumBlocksTune(IdxRange<T, T, T> idxRange) : Tuneable<T>(T(64), "gridSize", idxRange)
        {
        }

        // this should only be used to create a modified instance of an existing numBlockTune
        explicit NumBlocksTune(T initial_value, std::string const& name, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, name, idxRange)
        {
        }

        explicit NumBlocksTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, "gridSize", idxRange)
        {
        }

        void setGrid(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };

    template<typename T = alpaka::Vec<std::size_t, 1>>
    struct ThreadBlockSizeTune : public Tuneable<T>
    {
        T blockThreadSize;

        explicit ThreadBlockSizeTune() : Tuneable<T>(T(256), "threadBlockSize"), blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, "threadBlockSize", idxRange)
            , blockThreadSize(initial_value)
        {
        }

        // this should only be used to create a modified instance of an existing threadblockTune
        explicit ThreadBlockSizeTune(T initial_value, std::string const& name, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, name, idxRange)
            , blockThreadSize(initial_value)
        {
        }

        explicit ThreadBlockSizeTune(IdxRange<T, T, T> idxRange)
            : Tuneable<T>(T(256), "threadBlockSize", idxRange)
            , blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value)
            : Tuneable<T>(initial_value, "threadBlockSize")
            , blockThreadSize(initial_value)
        {
        }

        void setBlock(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };

    template<typename T = alpaka::Vec<std::size_t, 1>>
    struct NumFramesTune : public Tuneable<T>
    {
        explicit NumFramesTune() : Tuneable<T>(T(256), "numFrames")
        {
        }

        explicit NumFramesTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, "numFrames", idxRange)
        {
        }

        // this should only be used to create a modified instance of an existing threadblockTune
        explicit NumFramesTune(T initial_value, std::string const& name, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, name, idxRange)
        {
        }

        explicit NumFramesTune(IdxRange<T, T, T> idxRange) : Tuneable<T>(T(256), "numFrames", idxRange)
        {
        }

        explicit NumFramesTune(T initial_value) : Tuneable<T>(initial_value, "numFrames")
        {
        }

        void setBlock(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };

    template<typename T = alpaka::Vec<std::size_t, 1>>
    struct FrameExtentTune : public Tuneable<T>
    {
        explicit FrameExtentTune() : Tuneable<T>(T(256), "frameExtent")
        {
        }

        explicit FrameExtentTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, "frameExtent", idxRange)
        {
        }

        // this should only be used to create a modified instance of an existing threadblockTune
        explicit FrameExtentTune(T initial_value, std::string const& name, IdxRange<T, T, T> idxRange)
            : Tuneable<T>(initial_value, name, idxRange)
        {
        }

        explicit FrameExtentTune(IdxRange<T, T, T> idxRange) : Tuneable<T>(T(256), "frameExtent", idxRange)
        {
        }

        explicit FrameExtentTune(T initial_value) : Tuneable<T>(initial_value, "frameExtent")
        {
        }

        void setBlock(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };

} // namespace alpaka::tune

#endif // TUNEABLE_H
