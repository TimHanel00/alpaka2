//
// Created by tim on 23.04.25.
//
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP
#include <alpaka/tune/active/tuneable.hpp>

#include <tuple> // for std::tuple
#include <utility> // for std::move

template<typename T_KernelRun, auto Name>
constexpr auto getAccessorForName(T_KernelRun const& run)
{
    constexpr auto tag = T_KernelRun::template getName<Name>();

    if constexpr(tag != alpaka::tune::empty_name)
    {
        auto* ptr = run.template getByName<tag>();
        return std::make_tuple(std::ref(ptr->value)); // ✅ tuple of reference
    }
    else
        return std::tuple<>(); // ✅ empty if name doesn't exist
}

template<typename Environment, typename T_NamesTuple, typename T_KernelRun, std::size_t... Is>
constexpr auto constructAccessorTupleImpl(T_KernelRun const& run, std::index_sequence<Is...>)
{
    constexpr auto names = T_NamesTuple{}; // Value instance to extract from
    return std::tuple_cat(getAccessorForName<T_KernelRun, std::get<Is>(names)>(run)...);
}

template<typename Environment, typename T_KernelRun, auto... NameTags>
constexpr auto constructAccessorTuple(T_KernelRun const& run)
{
    return std::tuple_cat(getAccessorForName<T_KernelRun, NameTags>(run)...);
}

template<typename T_Predicate, StaticString... Names>
struct Constraint
{
    static constexpr std::tuple<decltype(Names)...> names{Names...};
    using PredicateType = T_Predicate;

    PredicateType predicate;

    constexpr Constraint(PredicateType p) : predicate(std::move(p))
    {
    }

    template<typename Environment, typename KernelRun>
    bool operator()(KernelRun const& run)
    {
        std::cout << " run into constraint " << std::endl;
        static auto accessorTuple = constructAccessorTuple<Environment, KernelRun, Names...>(run);
        //@TODO: this has to be done in the environment construction phase T_Strategy also has to belong inside their.
        return std::apply(predicate, accessorTuple);
    }
};
#endif // CONSTRAINT_HPP
