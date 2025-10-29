//
// Created by tim on 04.03.25.
//

#ifndef TUPLEHANDLE_H
#define TUPLEHANDLE_H
#include "alpaka/tune/concepts.hpp"
#include "alpaka/tune/traits/traits.hpp"

#include <string>
#include <vector>

namespace alpaka::tune::detail
{

    template<tune::concepts::Serializable T>
    std::string toStringGeneric(T const& value)
    {
        if constexpr(concepts::serialize::HasTraitSerializer<T>)
        {
            return ::alpaka::tune::trait::Serialize<T>{}(value);
        }
        else if constexpr(std::is_convertible_v<T, std::string>)
        {
            return std::string(value);
        }
        else if constexpr(std::is_arithmetic_v<T> || concepts::serialize::HasStdToString<T>)
        {
            return std::to_string(value);
        }
        else if constexpr(concepts::serialize::HasToStringMethod<T>)
        {
            return value.toString();
        }
        else if constexpr(concepts::serialize::HasStreamOperator<T>)
        {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }
        else
        {
            static_assert(
                sizeof(T) == 0,
                "Session specifier need to be convertible to string or missing alpaka::tune::Serialize<T> "
                "specialization");
        }
        return "";
    }

    inline void processArgs(std::vector<std::string>&)
    {
    }

    // Recursive case: contains arithmetric Argument
    template<typename First, typename... Rest>
    void processArgs(std::vector<std::string>& specifierStrings, First first, Rest... rest)
    {
        specifierStrings.push_back(toStringGeneric(first)); // Convert numbers to strings
        processArgs(specifierStrings, rest...); // Process remaining args
    }

    // Recursive case: contains string Argument
    template<typename... Args>
    void processArgs(std::vector<std::string>& specifierStrings, std::string const& str, Args... rest)
    {
        specifierStrings.push_back(str);
        processArgs(specifierStrings, rest...);
    }
} // namespace alpaka::tune::detail


#endif // TUPLEHANDLE_H
