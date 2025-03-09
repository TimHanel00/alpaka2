//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include "alpaka/mem/IdxRange.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
namespace alpaka::tune
{

// Placeholder for IdxRange

    // Interface for tunable objects
    inline std::size_t globalId=0;
    // Generic Tuneable class template
    template <typename T = std::size_t,
              typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct Tuneable {
        T value;
        std::string name;
        bool userDef;
        IdxRange<T_Begin, T_End, T_Stride> idxRange;

        // Helper function to provide a default IdxRange
        static IdxRange<T_Begin, T_End, T_Stride> defaultIdxRange(T value) {
            return IdxRange<T_Begin, T_End, T_Stride>{T_Begin{0}, T_End{value}, T_Stride{1}};
        }
        explicit Tuneable()
            : value(0),
              name("Tuneable: default"),
              userDef(false),
              idxRange(defaultIdxRange(0)) {}
        explicit Tuneable(T value)
            : value(value),
              name("Tuneable: " + std::to_string(globalId++)),
              userDef(false),
              idxRange(defaultIdxRange(value)) {}
        // Constructor with value and idxRange
        explicit Tuneable(T value, IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : value(value),
              name("Tuneable: " + std::to_string(globalId++)),
              userDef(true),
              idxRange(idxRange) {}

        // Constructor with idxRange only
        explicit Tuneable(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : name("Tuneable: " + std::to_string(globalId++)),
              userDef(true),
              idxRange(std::move(idxRange)) {
            value = (this->idxRange.m_end().product() - this->idxRange.m_begin().product()) / T(2);
        }

        // Constructor with idxRange and custom name
        explicit Tuneable(const std::string& name,IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : name(name),
              userDef(true),
              idxRange(std::move(idxRange)) {
            value = (this->idxRange.m_end().product() - this->idxRange.m_begin().product()) / T(2);
        }

        // Constructor with value and custom name
        explicit Tuneable(T value, const std::string& name)
            : value(value),
              name(name),
              userDef(false),
              idxRange(defaultIdxRange(value)) {}

        // Constructor with value, custom name, and idxRange
        explicit Tuneable(T value, const std::string& name, IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : value(value),
              name(name),
              userDef(true),
              idxRange(std::move(idxRange)) {}

        // Getter for name
        std::string getName() const { return name; }
        bool operator==(const Tuneable& other) const {
            return value == other.value &&
                   name == other.name;
        }
        //non default exlicit copy method to avoid accidently copying with = operator
        auto copy()
        {
            return Tuneable(value,name,idxRange);
        }
    };

    // Specialized tunables
    template <typename T = std::size_t,typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct GridSizeTune : Tuneable<T> {
        T gridSize;
        explicit GridSizeTune()
            : gridSize(T(64)),Tuneable<T>(T(64), "gridSize"){}
        explicit GridSizeTune(T initial_value)
            : gridSize(initial_value),Tuneable<T>(initial_value, "gridSize"){}
        explicit GridSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : gridSize(T(64)),Tuneable<T>(T(64), "gridSize", idxRange){}
        explicit GridSizeTune(T initial_value,
                              IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : gridSize(initial_value),Tuneable<T>(initial_value, "gridSize", idxRange){}
        void setGrid(const IdxRange<T_Begin, T_End, T_Stride> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };

    template <typename T = std::size_t,typename T_End = alpaka::Vec<T,1u>,
              typename T_Begin = alpaka::Vec<T,1u>,
              typename T_Stride = alpaka::Vec<T,1u>>
    struct ThreadBlockSizeTune : Tuneable<T> {
        T blockThreadSize;
        explicit ThreadBlockSizeTune()
            : blockThreadSize(T(256)),Tuneable<T>(T(256), "gridSize"){}
        explicit ThreadBlockSizeTune(T initial_value,IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : blockThreadSize(initial_value),Tuneable<T>(initial_value, "blockThreadSize", idxRange){}
        explicit ThreadBlockSizeTune(IdxRange<T_Begin, T_End, T_Stride> idxRange)
            : blockThreadSize(T(256)),Tuneable<T>(T(256), "blockThreadSize", idxRange){}
        explicit ThreadBlockSizeTune(T initial_value)
            : blockThreadSize(initial_value),Tuneable<T>(initial_value, "blockThreadSize"){}
        void setBlock(const IdxRange<T_Begin, T_End, T_Stride> &idxRange)
        {
            this->idxRange=std::optional<IdxRange<T_Begin, T_End, T_Stride>>(idxRange);
        }
    };


}

#endif //TUNEABLE_H
