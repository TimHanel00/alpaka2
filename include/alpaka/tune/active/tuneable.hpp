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

template<typename T_KernelRun, typename T_FrameSpec>
static T_FrameSpec& applyCustomThreadSpec(T_KernelRun& kernelRun, T_FrameSpec& spec)
{
    if(kernelRun.gridSize != std::nullopt)
    {
        auto val = kernelRun.gridSize->value;
        spec.m_threadSpec.m_numBlocks = ALPAKA_TYPEOF(spec.m_threadSpec.m_numBlocks)(val);
    }
    if(kernelRun.threadBlockSize != std::nullopt)
    {
        auto val = kernelRun.threadBlockSize->value;
        spec.m_threadSpec.m_numThreads = ALPAKA_TYPEOF(spec.m_threadSpec.m_numThreads)(val);
    }
    return spec;
}

namespace alpaka::tune
{
    void adjustToRange(auto& value, auto& begin, auto& end, auto& stride)
    {
        auto val = value;

        auto offset = val - begin;
        auto remainder = offset % stride;

        if(remainder == 0 && val >= begin && val <= end)
            return;
        std::cout << " Warning tuning Parameter does not meet the criteria specified by its Range! \n It will be "
                     "automatically adjusted."
                  << std::endl;
        auto n = offset / stride;
        if(remainder * 2 >= stride)
        {
            ++n; // Round up if closer
        }

        auto corrected = begin + n * stride;

        // Clamp within range
        if(corrected < begin)
            corrected = begin;
        if(corrected >= end)
            corrected = end - stride;

        value = corrected;
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
        IdxRange<alpaka::Vec<T, 1>, alpaka::Vec<T, 1>, alpaka::Vec<T, 1>> idxRange;

        FlatTuneableHandle(T& val, std::string const& n, bool u, T& b, T& e, T& s)
            : value(val)
            , name(n)
            , userDef(u)
            , idxRange(alpaka::Vec<T, 1>{b}, alpaka::Vec<T, 1>{e}, alpaka::Vec<T, 1>{s})
        {
        }

        T& value;
    };

    inline std::size_t globalId = 0;

    template<
        typename T,
        typename T_Begin = alpaka::Vec<T, 1>,
        typename T_End = alpaka::Vec<T, 1>,
        typename T_Stride = alpaka::Vec<T, 1>>
    struct Tuneable
    {
    };

    template<typename T, typename T_Begin, typename T_End, typename T_Stride>
    requires IsIntegral<T>
    struct Tuneable<T, T_Begin, T_End, T_Stride>
    {
        static inline int globalId = 0;

        T value;
        std::string name;
        bool userDef;
        IdxRange<T_Begin, T_End, T_Stride> idxRange;

        std::size_t numSteps()
        {
            return (idxRange.distance() / idxRange.m_stride) + 1;
        }

        std::vector<T> getValues()
        {
            return {value};
        }

        static IdxRange<T_Begin, T_End, T_Stride> defaultIdxRange(T val)
        {
            return IdxRange<T_Begin, T_End, T_Stride>{T_Begin(0), T_End(val), T_Stride(1)};
        }

        explicit Tuneable() : value(0), name("Tuneable: "), userDef(false), idxRange(defaultIdxRange(0))
        {
        }

        explicit Tuneable(T val) : value(val), name("Tuneable: "), userDef(false), idxRange(defaultIdxRange(val))
        {
        }

        // value and idxRange
        explicit Tuneable(T val, IdxRange<T_Begin, T_End, T_Stride> ir)
            : value(val)
            , name("Tuneable: ")
            , userDef(true)
            , idxRange(ir)
        {
            adjustToRange(value, this->idxRange.m_begin, this->idxRange.m_end, this->idxRange.m_stride);
        }

        // idxRange only
        explicit Tuneable(IdxRange<T_Begin, T_End, T_Stride> ir)
            : name("Tuneable: ")
            , userDef(true)
            , idxRange(std::move(ir))
        {
            // Arbitrary example guess for default 'value'

            value = (this->idxRange.m_end().product() - this->idxRange.m_begin.product()) / T(2);
            adjustToRange(value, this->idxRange.m_begin, this->idxRange.m_end, this->idxRange.m_stride);
        }

        // idxRange + custom name
        explicit Tuneable(std::string iname, IdxRange<T_Begin, T_End, T_Stride> ir)
            : name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            value = (this->idxRange.m_end().product() - this->idxRange.m_begin.product()) / T(2);
            adjustToRange(value, this->idxRange.m_begin, this->idxRange.m_end, this->idxRange.m_stride);
        }

        // value + custom name
        explicit Tuneable(T val, std::string iname)
            : value(val)
            , name(std::move(iname))
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {
        }

        // value + custom name + idxRange
        explicit Tuneable(T val, std::string iname, IdxRange<T_Begin, T_End, T_Stride> ir)
            : value(val)
            , name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            adjustToRange(value, this->idxRange.m_begin, this->idxRange.m_end(), this->idxRange.m_stride);
        }

        [[nodiscard]] std::string valueToString() const
        {
            return std::to_string(this->value);
        }

        [[nodiscard]] std::string toHash() const
        {
            return name + "*" + valueToString();
        }

        bool operator==(Tuneable const& other) const
        {
            return (value == other.value) && (name == other.name);
        }

        // Avoid accidental copying
        auto copy()
        {
            return Tuneable(value, name, idxRange);
        }
    };

    template<typename T>
    requires alpaka::isVector_v<T>
    struct Tuneable<T, T, T, T>
    {
        T value;
        std::string name;
        bool userDef;
        IdxRange<T, T, T> idxRange;

        // For a vector, define a default range that goes from "0" to "value",
        // and stride "1" in every dimension. For real code, adapt as needed.
        static IdxRange<T, T, T> defaultIdxRange(T const& val)
        {
            // Here we assume you can default-construct a vector of zeros
            // and a vector of "1" for stride. Or you might do something else.
            T zero{};
            T ones{};
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                zero[i] = 0;
                ones[i] = 1;
            }
            return IdxRange<T, T, T>{zero, val, ones};
        }

        std::size_t numSteps()
        {
            std::size_t numSteps = ((idxRange.m_end[0] - idxRange.m_begin[0]) / idxRange.m_stride[0]) + 1;
            for(std::size_t i = 1; i < alpaka::getDim(T{}); ++i)
            {
                numSteps = numSteps * (((idxRange.m_end[i] - idxRange.m_begin[i]) / idxRange.m_stride[i]) + 1);
            }
            return numSteps;
        }

        std::vector<typename T::type> getValues()
        {
            std::vector<typename T::type> ret;
            for(auto i = 0; i < alpaka::getDim(T{}); ++i)
            {
                ret.emplace_back(value[i]);
            }
            return ret;
        }

        // Constructors mimic the integral version, but with T => alpaka::Vec.
        explicit Tuneable() : value{}, name("TuneableVec: default"), userDef(false), idxRange(defaultIdxRange(T{}))
        {
        }

        explicit Tuneable(T const& val)
            : value(val)
            , name("TuneableVec: ")
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {
        }

        explicit Tuneable(T const& val, IdxRange<T, T, T> ir)
            : value(val)
            , name("TuneableVec: ")
            , userDef(true)
            , idxRange(ir)
        {
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                adjustToRange(
                    value[i],
                    this->idxRange.m_begin[i],
                    this->idxRange.m_end[i],
                    this->idxRange.m_stride[i]);
            }
        }

        explicit Tuneable(IdxRange<T, T, T> ir) : name("TuneableVec: "), userDef(true), idxRange(std::move(ir))
        {
            // Example logic: set 'value' = (end - begin) / 2 in each dimension
            T half{};
            for(auto i = 0; i < alpaka::getDim(T{}); ++i)
            {
                half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
            }
            value = half;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                adjustToRange(
                    value[i],
                    this->idxRange.m_begin[i],
                    this->idxRange.m_end[i],
                    this->idxRange.m_stride[i]);
            }
        }

        explicit Tuneable(std::string iname, IdxRange<T, T, T> ir)
            : name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            T half{};
            for(auto i = 0; i < alpaka::getDim(T{}); ++i)
            {
                half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
            }
            value = half;
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                adjustToRange(
                    value[i],
                    this->idxRange.m_begin[i],
                    this->idxRange.m_end[i],
                    this->idxRange.m_stride[i]);
            }
        }

        explicit Tuneable(T const& val, std::string iname)
            : value(val)
            , name(std::move(iname))
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {
        }

        explicit Tuneable(T const& val, std::string iname, IdxRange<T, T, T> ir)
            : value(val)
            , name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            for(std::size_t i = 0; i < alpaka::getDim(T{}); ++i)
            {
                adjustToRange(
                    value[i],
                    this->idxRange.m_begin[i],
                    this->idxRange.m_end[i],
                    this->idxRange.m_stride[i]);
            }
        }

        std::string toHash() const
        {
            return name + "*" + value.toString();
        }

        bool operator==(Tuneable const& other) const
        {
            // For vector equality, you'd compare each component, or rely
            // on operator== if your library provides it for alpaka::Vec.
            return (value == other.value) && (name == other.name);
        }

        auto copy()
        {
            return Tuneable(value, name, idxRange);
        }
    };

    template<typename T>
    requires alpaka::isVector_v<T>
    Tuneable(T) -> Tuneable<T, T, T, T>;

    // Specialized tunables


    template<
        typename T = std::size_t,
        typename T_End = alpaka::Vec<T, 1u>,
        typename T_Begin = alpaka::Vec<T, 1u>,
        typename T_Stride = alpaka::Vec<T, 1u>>
    struct GridSizeTune : Tuneable<T, T_Begin, T_End, T_Stride>
    {
        T gridSize;

        explicit GridSizeTune() : Tuneable<T, T_Begin, T_End, T_Stride>(T(64), "gridSize"), gridSize(T(64))
        {
        }

        explicit GridSizeTune(T initial_value)
            : Tuneable<T, T_Begin, T_End, T_Stride>(initial_value, "gridSize")
            , gridSize(initial_value)
        {
        }

        explicit GridSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T, T_Begin, T_End, T_Stride>(T(64), "gridSize", idxRange)
            , gridSize(T(64))
        {
        }

        explicit GridSizeTune(T initial_value, IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T, T_Begin, T_End, T_Stride>(initial_value, "gridSize", idxRange)
            , gridSize(initial_value)
        {
        }

        void setGrid(IdxRange<T_Begin, T_End, T_Stride> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };

    template<typename T>
    requires alpaka::isVector_v<T>
    struct GridSizeTune<T, T, T, T> : Tuneable<T, T, T, T>
    {
        T gridSize;

        explicit GridSizeTune() : Tuneable<T, T, T, T>(T(64), "gridSize"), gridSize(T(64))
        {
        }

        explicit GridSizeTune(T initial_value)
            : Tuneable<T, T, T, T>(initial_value, "gridSize")
            , gridSize(initial_value)
        {
        }

        explicit GridSizeTune(IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(T(64), "gridSize", idxRange)
            , gridSize(T(64))
        {
        }

        explicit GridSizeTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(initial_value, "gridSize", idxRange)
            , gridSize(initial_value)
        {
        }

        void setGrid(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };

    template<
        typename T = std::size_t,
        typename T_End = alpaka::Vec<T, 1u>,
        typename T_Begin = alpaka::Vec<T, 1u>,
        typename T_Stride = alpaka::Vec<T, 1u>>
    struct ThreadBlockSizeTune : Tuneable<T, T_Begin, T_End, T_Stride>
    {
        T blockThreadSize;

        explicit ThreadBlockSizeTune()
            : Tuneable<T, T_Begin, T_End, T_Stride>(T(256), "blockThreadSize")
            , blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value, IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T, T_Begin, T_End, T_Stride>(initial_value, "blockThreadSize", idxRange)
            , blockThreadSize(initial_value)
        {
        }

        explicit ThreadBlockSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T, T_Begin, T_End, T_Stride>(T(256), "blockThreadSize", idxRange)
            , blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value)
            : Tuneable<T, T_Begin, T_End, T_Stride>(initial_value, "blockThreadSize")
            , blockThreadSize(initial_value)
        {
        }

        void setBlock(IdxRange<T_Begin, T_End, T_Stride> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };

    template<typename T>
    requires alpaka::isVector_v<T>
    struct ThreadBlockSizeTune<T, T, T, T> : Tuneable<T, T, T, T>
    {
        T blockThreadSize;

        explicit ThreadBlockSizeTune() : Tuneable<T, T, T, T>(T(256), "blockThreadSize"), blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value, IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(initial_value, "blockThreadSize", idxRange)
            , blockThreadSize(initial_value)
        {
        }

        explicit ThreadBlockSizeTune(IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(T(256), "blockThreadSize", idxRange)
            , blockThreadSize(T(256))
        {
        }

        explicit ThreadBlockSizeTune(T initial_value)
            : Tuneable<T, T, T, T>(initial_value, "blockThreadSize")
            , blockThreadSize(initial_value)
        {
        }

        void setBlock(IdxRange<T, T, T> const& idxRange)
        {
            this->idxRange = std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };


} // namespace alpaka::tune

#endif // TUNEABLE_H
