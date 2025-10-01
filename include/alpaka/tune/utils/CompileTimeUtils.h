//
// Created by tim on 06.05.25.
//

#ifndef COMPILETIMEUTILS_H
#define COMPILETIMEUTILS_H
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

namespace alpaka::tune::utils
{
    template<auto N, typename Func, std::size_t... Is>
    constexpr void unroll_impl(Func&& f, std::index_sequence<Is...>)
    {
        (f(std::integral_constant<decltype(N), Is>{}), ...);
    }

    template<auto N, typename Func>
    constexpr void unroll(Func&& f)
    {
        static_assert(N >= 0, "Unroll count must be non-negative");
        unroll_impl<N>(std::forward<Func>(f), std::make_index_sequence<N>{});
    }
} // namespace alpaka::tune::utils
#endif // COMPILETIMEUTILS_H
