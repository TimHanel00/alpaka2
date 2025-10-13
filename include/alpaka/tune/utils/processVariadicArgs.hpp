//
// Created by tim on 04.03.25.
//

#ifndef TUPLEHANDLE_H
#define TUPLEHANDLE_H
#include <string>
#include <vector>

namespace alpaka::tune::detail
{
    inline void processArgs(std::vector<std::string>&)
    {
    }

    // Recursive case: contains arithmetric Argument
    template<typename First, typename... Rest>
    void processArgs(std::vector<std::string>& specifierStrings, First first, Rest... rest)
    {
        if constexpr(std::is_arithmetic_v<First>)
        {
            specifierStrings.push_back(std::to_string(first)); // Convert numbers to strings
        }
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
