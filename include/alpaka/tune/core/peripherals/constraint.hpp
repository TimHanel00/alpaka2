//
// Created by tim on 23.04.25.
//
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP
#include "alpaka/tune/tuneable/kernelTuningModel.hpp"

#include <iostream>
#include <tuple> // for std::tuple
#include <utility> // for std::move

template<auto ID, typename Accessor>
constexpr auto make_accessor_if_id_matches(Accessor const& acc)
{
    using A = std::remove_reference_t<Accessor>;
    if constexpr(A::ID == ID)
        return std::tuple{acc.m_value};
    else
        return std::tuple{};
}

// ===========================================================
//  Main: search tuple for a ParameterAccessor with given ID
// ===========================================================
template<typename T_ParameterTuple, auto ID>
constexpr auto getAccessorForID(T_ParameterTuple const& accessorTuple)
{
    using TupleType = std::remove_reference_t<T_ParameterTuple>;
    constexpr std::size_t N = std::tuple_size_v<TupleType>;

    return [&]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        // Flatten all one-element-or-empty tuples into one combined tuple
        return std::tuple_cat(make_accessor_if_id_matches<ID>(std::get<Is>(accessorTuple))...);
    }(std::make_index_sequence<N>{});
}

template<typename T_ParameterAccessor, auto... IDs>
constexpr auto constructAccessorTuple(T_ParameterAccessor const& run)
{
    return std::tuple_cat(getAccessorForID<T_ParameterAccessor, IDs>(run)...);
}

template<typename T_Predicate, auto... IDs>
struct Constraint
{
    template<typename... Args>
    bool operator()(std::tuple<ParameterAccessor<Args...>> const& accessor) const
    {
        // Build a tuple of accessors matching the compile-time ID pack
        auto accessorTuple = constructAccessorTuple<std::tuple<ParameterAccessor<Args...>>, IDs...>(
            const_cast<std::tuple<ParameterAccessor<Args...>>&>(accessor));

        return std::apply(T_Predicate{}, accessorTuple);
    }
};

#endif // CONSTRAINT_HPP
