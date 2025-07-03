//
// Created by tim on 03.07.25.
//

#ifndef TUPLEHELPER_H
#define TUPLEHELPER_H
#include <cstddef> // for std::size_t
#include <tuple> // for std::tuple, std::get, std::tuple_size, etc.
#include <type_traits>
#include <utility> // for std::index_sequence, std::make_index_sequence, std::forward

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

template<typename Tuple, typename F, std::size_t... Is>
void for_each_enumerate_impl(Tuple&& tup, F&& f, std::index_sequence<Is...>)
{
    (f(std::get<Is>(tup), Is), ...);
}

template<typename Tuple, typename F>
void for_each_enumerate(Tuple&& tup, F&& f)
{
    constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    for_each_enumerate_impl(std::forward<Tuple>(tup), std::forward<F>(f), std::make_index_sequence<N>{});
}

template<std::size_t I = 0, typename Tuple, typename Func>
void visitIndex(std::size_t i, Tuple&& tuple, Func&& f)
{
    if constexpr(I < std::tuple_size_v<std::remove_reference_t<Tuple>>)
    {
        if(i == I)
        {
            f(std::get<I>(std::forward<Tuple>(tuple)));
        }
        else
        {
            visitIndex<I + 1>(i, std::forward<Tuple>(tuple), std::forward<Func>(f));
        }
    }
    else
    {
        throw std::out_of_range("visitIndex: index out of bounds");
    }
}
#endif // TUPLEHELPER_H
