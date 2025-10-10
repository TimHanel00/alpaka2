//
// Created by tim on 03.07.25.
//

#ifndef TUPLEHELPER_H
#define TUPLEHELPER_H
#include <cstddef> // for std::size_t
#include <tuple> // for std::tuple, std::get, std::tuple_size, etc.
#include <type_traits>
#include <utility> // for std::index_sequence, std::make_index_sequence, std::forward

namespace alpaka::tune::utils
{

    template<typename T>
    struct is_empty_tuple : std::false_type
    {
    };

    template<>
    struct is_empty_tuple<std::tuple<>> : std::true_type
    {
    };

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
    constexpr void for_each_enumerate_constexpr_impl(Tuple&& tup, F&& f, std::index_sequence<Is...>)
    {
        (f.template operator()<Is>(std::get<Is>(tup)), ...);
    }

    // Entry point
    template<typename Tuple, typename F, std::size_t... Is>
    constexpr void for_each_enumerate_runtime_impl(Tuple&& tup, F&& f, std::index_sequence<Is...>)
    {
        (f(std::get<Is>(tup), Is), ...);
    }

    template<typename Tuple, typename F>
    constexpr void for_each_enumerate(Tuple&& tup, F&& f)
    {
        if constexpr(requires { f.template operator()<0>(std::get<0>(tup)); })
        {
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
            for_each_enumerate_constexpr_impl(
                std::forward<Tuple>(tup),
                std::forward<F>(f),
                std::make_index_sequence<N>{});
        }
        else
        {
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
            for_each_enumerate_runtime_impl(
                std::forward<Tuple>(tup),
                std::forward<F>(f),
                std::make_index_sequence<N>{});
        }
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
} // namespace alpaka::tune::utils
#endif // TUPLEHELPER_H
