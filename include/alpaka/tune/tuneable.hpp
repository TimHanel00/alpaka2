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
namespace alpaka::tune
{

    struct StorageTuneable
    {
        std::string name;
        std::string value;
        [[nodiscard]] std::string toHash() const
        {
            return name+"*"+value;
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
    struct FlatTuneableHandle {
        ;
        std::string name;
        bool userDef;
        IdxRange<alpaka::Vec<T,1>,alpaka::Vec<T,1>,alpaka::Vec<T,1>> idxRange;

        FlatTuneableHandle(T& val, const std::string& n, bool u, T &b, T &e, T &s)
            : value(val), name(n), userDef(u), idxRange(alpaka::Vec<T,1>{b}, alpaka::Vec<T,1>{e}, alpaka::Vec<T,1>{s}) {}
        T& value;
    };
    inline std::size_t globalId=0;
    template<
    typename T,
    typename T_Begin  = alpaka::Vec<T,1>,
    typename T_End    = alpaka::Vec<T,1>,
    typename T_Stride = alpaka::Vec<T,1>>
    struct Tuneable
    {};
    template<typename T, typename T_Begin, typename T_End, typename T_Stride>
    requires IsIntegral<T>
    struct Tuneable<T, T_Begin, T_End, T_Stride>
    {
        static inline int globalId = 0;

        T value;
        std::string name;
        bool userDef;
        IdxRange<T_Begin, T_End, T_Stride> idxRange;

        static IdxRange<T_Begin, T_End, T_Stride> defaultIdxRange(T val)
        {
            return IdxRange<T_Begin, T_End, T_Stride>
            {
                T_Begin(0),
                T_End(val),
                T_Stride(1)
            };
        }

        constexpr explicit Tuneable()
            : value(0)
            , name("Tuneable: ")
            , userDef(false)
            , idxRange(defaultIdxRange(0))
        {}

        constexpr explicit Tuneable(T val)
            : value(val)
            , name("Tuneable: ")
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {}

        // value and idxRange
        constexpr explicit Tuneable(T val, IdxRange<T_Begin, T_End, T_Stride> ir)
            : value(val)
            , name("Tuneable: ")
            , userDef(true)
            , idxRange(ir)
        {}

        // idxRange only
        constexpr explicit Tuneable(IdxRange<T_Begin, T_End, T_Stride> ir)
            : name("Tuneable: ")
            , userDef(true)
            , idxRange(std::move(ir))
        {
            // Arbitrary example guess for default 'value'
            value = (this->idxRange.m_end().product() - this->idxRange.m_begin().product()) / T(2);
        }

        // idxRange + custom name
        constexpr explicit Tuneable(std::string iname, IdxRange<T_Begin, T_End, T_Stride> ir)
            : name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {
            value = (this->idxRange.m_end().product() - this->idxRange.m_begin().product()) / T(2);
        }

        // value + custom name
        constexpr explicit Tuneable(T val, std::string iname)
            : value(val)
            , name(std::move(iname))
            , userDef(false)
            , idxRange(defaultIdxRange(val))
        {}

        // value + custom name + idxRange
        constexpr explicit Tuneable(T val, std::string iname, IdxRange<T_Begin, T_End, T_Stride> ir)
            : value(val)
            , name(std::move(iname))
            , userDef(true)
            , idxRange(std::move(ir))
        {}
        [[nodiscard]] std::string valueToString() const { return std::to_string(this->value); }
        [[nodiscard]] std::string toHash() const { return name+"*"+valueToString();}

        bool operator==(const Tuneable& other) const
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
    static IdxRange<T, T, T> defaultIdxRange(const T &val)
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
        return IdxRange<T, T, T>{ zero, val, ones };
    }

    // Constructors mimic the integral version, but with T => alpaka::Vec.
    constexpr explicit Tuneable()
        : value{}
        , name("TuneableVec: default")
        , userDef(false)
        , idxRange(defaultIdxRange(T{}))
    {}

    constexpr explicit Tuneable(const T &val)
        : value(val)
        , name("TuneableVec: ")
        , userDef(false)
        , idxRange(defaultIdxRange(val))
    {}

    constexpr explicit Tuneable(const T &val, IdxRange<T, T, T> ir)
        : value(val)
        , name("TuneableVec: ")
        , userDef(true)
        , idxRange(ir)
    {}

    constexpr explicit Tuneable(IdxRange<T, T, T> ir)
        : name("TuneableVec: ")
        , userDef(true)
        , idxRange(std::move(ir))
    {
        // Example logic: set 'value' = (end - begin) / 2 in each dimension
        T half{};
        for(auto i = 0; i < alpaka::getDim(T{}); ++i)
        {
            half[i] = (idxRange.m_end()[i] - idxRange.m_begin()[i]) / 2;
        }
        value = half;
    }

    constexpr explicit Tuneable(std::string iname, IdxRange<T, T, T> ir)
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
    }

    constexpr explicit Tuneable(const T &val, std::string iname)
        : value(val)
        , name(std::move(iname))
        , userDef(false)
        , idxRange(defaultIdxRange(val))
    {}

    constexpr explicit Tuneable(const T &val, std::string iname, IdxRange<T, T, T> ir)
        : value(val)
        , name(std::move(iname))
        , userDef(true)
        , idxRange(std::move(ir))
    {}
        std::string toHash() const { return name+"*"+value.toString(); }



    bool operator==(const Tuneable& other) const
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


    template <typename T = std::size_t,
          typename T_End = alpaka::Vec<T,1u>,
          typename T_Begin = alpaka::Vec<T,1u>,
          typename T_Stride = alpaka::Vec<T,1u>>
    struct GridSizeTune : Tuneable<T, T_Begin, T_End, T_Stride> {
        T gridSize;
        constexpr explicit GridSizeTune()
            : Tuneable<T,T_Begin,T_End,T_Stride>(T(64), "gridSize"), gridSize(T(64))
        {}
        constexpr explicit GridSizeTune(T initial_value)
            : Tuneable<T,T_Begin,T_End,T_Stride>(initial_value, "gridSize"), gridSize(initial_value)
        {}
        constexpr explicit GridSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T,T_Begin,T_End,T_Stride>(T(64), "gridSize", idxRange)
            , gridSize(T(64))
        {}
        constexpr explicit GridSizeTune(T initial_value,
                              IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T,T_Begin,T_End,T_Stride>(initial_value, "gridSize", idxRange)
            , gridSize(initial_value)
        {}
        constexpr void setGrid(const IdxRange<T_Begin, T_End, T_Stride> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };
    template<typename T>
    requires alpaka::isVector_v<T>
    struct GridSizeTune<T, T, T, T> : Tuneable<T, T, T, T> {
        T gridSize;
        constexpr explicit GridSizeTune()
            : Tuneable<T, T, T, T>(T(64), "gridSize"), gridSize(T(64))
        {}
        constexpr explicit GridSizeTune(T initial_value)
            : Tuneable<T, T, T, T>(initial_value, "gridSize"), gridSize(initial_value)
        {}
        constexpr explicit GridSizeTune(IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(T(64), "gridSize", idxRange)
            , gridSize(T(64))
        {}
        constexpr explicit GridSizeTune(T initial_value,
                              IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(initial_value, "gridSize", idxRange)
            , gridSize(initial_value)
        {}
        constexpr void setGrid(const IdxRange<T, T, T> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };
    template <typename T = std::size_t,typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct ThreadBlockSizeTune : Tuneable<T,T_Begin,T_End,T_Stride> {
        T blockThreadSize;
        constexpr explicit ThreadBlockSizeTune()
            : Tuneable<T,T_Begin,T_End,T_Stride>(T(256), "gridSize"), blockThreadSize(T(256))
        {}
        constexpr explicit ThreadBlockSizeTune(T initial_value,IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T,T_Begin,T_End,T_Stride>(initial_value, "blockThreadSize", idxRange)
            , blockThreadSize(initial_value)
        {}
        constexpr explicit ThreadBlockSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : Tuneable<T,T_Begin,T_End,T_Stride>(T(256), "blockThreadSize", idxRange)
            , blockThreadSize(T(256))
        {}
        constexpr explicit ThreadBlockSizeTune(T initial_value)
            : Tuneable<T,T_Begin,T_End,T_Stride>(initial_value, "blockThreadSize")
            , blockThreadSize(initial_value)
        {}
        constexpr void setBlock(const IdxRange<T_Begin, T_End, T_Stride> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };
    template <typename T>
    requires alpaka::isVector_v<T>
    struct ThreadBlockSizeTune<T, T, T, T> : Tuneable<T, T, T, T>  {
        T blockThreadSize;
        constexpr explicit ThreadBlockSizeTune()
            : Tuneable<T, T, T, T>(T(256), "gridSize"), blockThreadSize(T(256))
        {}
        constexpr explicit ThreadBlockSizeTune(T initial_value,IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(initial_value, "blockThreadSize", idxRange)
            , blockThreadSize(initial_value)
        {}
        constexpr explicit ThreadBlockSizeTune(IdxRange<T, T, T> idxRange)
            : Tuneable<T, T, T, T>(T(256), "blockThreadSize", idxRange)
            , blockThreadSize(T(256))
        {}
        constexpr explicit ThreadBlockSizeTune(T initial_value)
            : Tuneable<T, T, T, T>(initial_value, "blockThreadSize")
            , blockThreadSize(initial_value)
        {}
        constexpr void setBlock(const IdxRange<T, T, T> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T, T, T>>(idxRange);
        }
    };


}

#endif //TUNEABLE_H
