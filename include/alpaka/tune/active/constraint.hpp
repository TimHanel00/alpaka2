//
// Created by tim on 23.04.25.
//
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include "activeKernel.hpp"

#include <tuple> // for std::tuple
#include <utility> // for std::move

template<typename T_KernelRun, auto ID>
constexpr auto getAccessorForName(T_KernelRun& run)
{
    auto* ptr = run.template getByID<ID>();
    using T_tune = std::remove_cvref_t<ALPAKA_TYPEOF(*ptr)>;
    if constexpr(std::is_same_v<T_tune, alpaka::tune::NoTune>)
    {
        return std::make_tuple();
    }
    else
    {
        return std::make_tuple(std::ref(ptr->value));
    }
}

template<typename Environment, typename T_NamesTuple, typename T_KernelRun, std::size_t... Is>
constexpr auto constructAccessorTupleImpl(T_KernelRun& run, std::index_sequence<Is...>)
{
    constexpr auto names = T_NamesTuple{}; // Value instance to extract from
    return std::tuple_cat(getAccessorForName<T_KernelRun, std::get<Is>(names)>(run)...);
}

template<typename Environment, typename T_KernelRun, auto... IDs>
constexpr auto constructAccessorTuple(T_KernelRun& run)
{
    return std::tuple_cat(getAccessorForName<T_KernelRun, IDs>(run)...);
}

template<typename T_Predicate, auto... IDs>
struct Constraint
{
    static constexpr std::tuple<decltype(IDs)...> names{IDs...};
    using PredicateType = T_Predicate;

    PredicateType predicate;

    constexpr Constraint(PredicateType p) : predicate(std::move(p))
    {
    }

    template<typename Environment, typename KernelRun>
    bool operator()(KernelRun& run)
    {
        static auto accessorTuple = constructAccessorTuple<Environment, KernelRun, IDs...>(run);

        constexpr std::size_t expectedSize = sizeof...(IDs);
        constexpr std::size_t actualSize = std::tuple_size_v<decltype(accessorTuple)>;

        if constexpr(actualSize != expectedSize)
        {
            std::cerr << "[Constraint] Warning: Accessor tuple size (" << actualSize
                      << ") does not match expected number of IDs (" << expectedSize
                      << "). Skipping constraint evaluation. Tuneable IDs: ";

            // Print all IDs in one line, separated by commas
            bool first = true;
            ((std::cerr << (first ? "" : ", ") << IDs, first = false), ...);
            std::cerr << std::endl;

            return true;
        }
        else
        {
            return std::apply(predicate, accessorTuple);
        }
    }
};
#endif // CONSTRAINT_HPP
