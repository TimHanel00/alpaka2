//
// Created by tim on 06.05.25.
//

#ifndef COMPILETIMETEMPLATES_H

#define COMPILETIMETEMPLATES_H
#include <alpaka/Vec.hpp>
#include <alpaka/alpaka.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>
#include <alpaka/tune/traits/traits.hpp>
#include <alpaka/tune/utils/tupleHash.h>

namespace alpaka::tune
{
    namespace trait
    {
        template<typename T>
        struct is_empty_tuple : std::false_type
        {
        };

        template<>
        struct is_empty_tuple<std::tuple<>> : std::true_type
        {
        };

        // Primary template
        template<typename Kernel, typename = void>
        struct hasUserDefinedCTuneable : std::false_type
        {
        };

        // Specialization when tuneAbleDefinitions() is valid
        template<typename Kernel>
        struct hasUserDefinedCTuneable<
            Kernel,
            std::void_t<decltype(alpaka::tune::trait::CompileTimeTuneableTrait<Kernel>::tuneAbleDefinitions())>>
        {
        private:
            using TupleType = decltype(alpaka::tune::trait::CompileTimeTuneableTrait<Kernel>::tuneAbleDefinitions());

        public:
            static constexpr bool value = !is_empty_tuple<TupleType>::value;
        };
    } // namespace trait

    namespace CompileTimeHelpers
    {
        namespace expandFromDefinition
        {
            template<typename A, typename B>
            struct all_less_equal;

            template<typename T, T... As, T... Bs>
            struct all_less_equal<CVec<T, As...>, CVec<T, Bs...>> : std::bool_constant<((As <= Bs) && ...)>
            {
            };

            template<typename T, T... A, T... B>
            constexpr auto addCVec(CVec<T, A...> a, CVec<T, B...> b)
            {
                static_assert(sizeof...(A) == sizeof...(B));
                return CVec<T, (A + B)...>{};
            }

            template<typename Current, typename End, typename Stride, typename = void>
            struct GenerateRecursive;

            // Recursive case
            template<typename Current, typename End, typename Stride>
            struct GenerateRecursive<Current, End, Stride, std::enable_if_t<all_less_equal<Current, End>::value>>
            {
                using next = decltype(addCVec(Current{}, Stride{}));
                using tail = typename GenerateRecursive<next, End, Stride>::type;
                using type = decltype(std::tuple_cat(std::tuple<Current>{}, tail{}));
            };

            // Base case
            template<typename Current, typename End, typename Stride>
            struct GenerateRecursive<Current, End, Stride, std::enable_if_t<!all_less_equal<Current, End>::value>>
            {
                using type = std::tuple<>;
            };
            template<typename Tunable>
            struct UnwrappTuneableType;

            template<// e.g., CVec<T, ...>
                    typename Begin,
                    typename End,
                    typename Stride,
                    std::size_t ID>
                struct UnwrappTuneableType<CTunable<Begin, End, Stride, ID>>
            {
                using T = typename Begin::type;
                using type_ = typename GenerateRecursive<Begin, End, Stride>::type;
            };

            template<typename Tuple, std::size_t... Is>
            constexpr auto expandValuesFromDefinition(std::index_sequence<Is...>)
            {
                return std::tuple<typename UnwrappTuneableType<std::tuple_element_t<Is, Tuple>>::type_...>{};
            }

            template<typename Tuple>
            constexpr auto expandValuesFromDefinition()
            {
                return expandValuesFromDefinition<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
            }
        } // namespace expandFromDefinition

        namespace allCombinations
        {
            template<typename T, typename Tuple>
            struct tuple_prepend;

            template<typename T, typename... Ts>
            struct tuple_prepend<T, std::tuple<Ts...>>
            {
                using type = std::tuple<T, Ts...>;
            };
            template<typename T, typename TupleOfTuples>
            struct prepend_to_all;

            template<typename T, typename... Tuples>
            struct prepend_to_all<T, std::tuple<Tuples...>>
            {
                using type = std::tuple<typename tuple_prepend<T, Tuples>::type...>;
            };
            template<typename... Tuples>
            struct tuple_cat_t;

            template<>
            struct tuple_cat_t<>
            {
                using type = std::tuple<>;
            };

            template<typename... Ts>
            struct tuple_cat_t<std::tuple<Ts...>>
            {
                using type = std::tuple<Ts...>;
            };

            template<typename... T1s, typename... T2s, typename... Rest>
            struct tuple_cat_t<std::tuple<T1s...>, std::tuple<T2s...>, Rest...>
            {
                using type = typename tuple_cat_t<std::tuple<T1s..., T2s...>, Rest...>::type;
            };

            template<typename... Tuples>
            struct CartesianProduct;

            template<>
            struct CartesianProduct<>
            {
                using type = std::tuple<std::tuple<>>;
            };

            // Recursive case
            template<typename FirstTuple, typename... RestTuples>
            struct CartesianProduct<FirstTuple, RestTuples...>
            {
            private:
                using RestProduct = typename CartesianProduct<RestTuples...>::type;

                template<typename T>
                using PrependAll = typename prepend_to_all<T, RestProduct>::type;

                template<typename... Ts>
                static auto expand(std::tuple<Ts...>) -> typename tuple_cat_t<PrependAll<Ts>...>::type;

            public:
                using type = decltype(expand(std::declval<FirstTuple>()));
            };
            template<typename TupleOfTuples>
            struct CartesianFromTuple;

            template<typename... Tuples>
            struct CartesianFromTuple<std::tuple<Tuples...>>
            {
                using type = typename CartesianProduct<Tuples...>::type;
            };
        } // namespace allCombinations

        namespace createNewKernel
        {
            template<typename T>
            struct ExtractTemplateArgsFromGenericKernel;

            // Generic fallback
            template<typename T>
            struct ExtractTemplateArgsFromGenericKernel
            {
                using type = std::tuple<>;
            };

            // Specialization for any class template with type parameters
            template<template<typename...> class Template, typename... Args>
            struct ExtractTemplateArgsFromGenericKernel<Template<Args...>>
            {
                // using TemplateType = Template<>;
                using type = std::tuple<Args...>;
            };

            template<std::size_t I, typename VecT>
            struct index_in
            {
            private:
                // Convert the CVec-based Vec to an integer_sequence
                using Seq = decltype(::alpaka::detail::toIntegerSequence(std::declval<VecT>()));

                template<std::size_t Target, std::size_t Index, typename T, T... Elems>
                struct find_index_impl;

                template<std::size_t Target, std::size_t Index, typename T, T First, T... Rest>
                struct find_index_impl<Target, Index, T, First, Rest...>
                {
                    static constexpr std::size_t value
                        = First == Target ? Index : find_index_impl<Target, Index + 1, T, Rest...>::value;
                };

                // Proper base case specialization: empty pack
                template<std::size_t Target, std::size_t Index, typename T>
                struct find_index_impl<Target, Index, T>
                {
                    static constexpr std::size_t value = static_cast<std::size_t>(-1); // Not found
                };

                template<typename T, T... Elems>
                struct find_index
                {
                    static constexpr std::size_t value = find_index_impl<I, 0, T, Elems...>::value;
                };
                template<typename Sequence>
                struct get_index;

                template<typename T, T... Elems>
                struct get_index<std::integer_sequence<T, Elems...>>
                {
                    static constexpr std::size_t value = find_index<T, Elems...>::value;
                };

            public:
                static constexpr std::size_t index = get_index<Seq>::value;
                static constexpr bool value = index != static_cast<std::size_t>(-1);
            };
            template<std::size_t N>
            struct debug_print;

            template<
                typename Tuple,
                typename Indices,
                typename Replacements,
                std::size_t CurrentKernelIndex,
                std::size_t NumberOfKernelArgs,
                bool Done,
                typename... Result>
            struct rebuild_tuple_impl;

            template<
                typename Tuple,
                typename Indices,
                typename Replacements,
                std::size_t CurrentKernelIndex,
                std::size_t NumberOfKernelArgs,
                typename... Result>
            struct rebuild_tuple_impl<
                Tuple,
                Indices,
                Replacements,
                CurrentKernelIndex,
                NumberOfKernelArgs,
                false,
                Result...>
            {
                using current = std::tuple_element_t<CurrentKernelIndex, Tuple>;
                using index_in_T = index_in<CurrentKernelIndex, std::remove_cvref_t<Indices>>;
                // static_assert(std::is_same_v<index_in_T, void()>);
                static constexpr std::size_t indexWhereCurrentKernelIndexWasFound
                    = (index_in_T::index == static_cast<std::size_t>(-1)) ? 0 : index_in_T::index;
                using replacement = std::tuple_element_t<indexWhereCurrentKernelIndexWasFound, Replacements>;
                static constexpr bool typeMatches = std::is_same_v<typename replacement::type, typename current::type>;

                static constexpr bool dimensionMatches
                    = ::alpaka::getDim(current{}) == ::alpaka::getDim(replacement{});
                static_assert(!index_in_T::value || typeMatches, " not type convertible");
                static_assert(
                    !index_in_T::value || (index_in_T::value && typeMatches && dimensionMatches),
                    " Template arguments of your Kernel definition do not match the corresponding compile time "
                    "tuneable definition (check types and dimensions)");
                static constexpr bool typeReturned = index_in_T::value && typeMatches && dimensionMatches;
                using type = typename std::conditional_t<
                    typeReturned,
                    rebuild_tuple_impl<
                        Tuple,
                        Indices,
                        Replacements,
                        CurrentKernelIndex + 1,
                        NumberOfKernelArgs,
                        (CurrentKernelIndex + 1 >= NumberOfKernelArgs),
                        Result...,
                        std::tuple_element_t<indexWhereCurrentKernelIndexWasFound, Replacements>>,
                    rebuild_tuple_impl<
                        Tuple,
                        Indices,
                        Replacements,
                        CurrentKernelIndex + 1,
                        NumberOfKernelArgs,
                        (CurrentKernelIndex + 1 >= NumberOfKernelArgs),
                        Result...,
                        current>>::type;
            };

            template<
                typename Tuple,
                typename Indices,
                typename Replacements,
                std::size_t CurrentKernelIndex,
                std::size_t NumberOfKernelArgs,
                typename... Result>
            struct rebuild_tuple_impl<
                Tuple,
                Indices,
                Replacements,
                CurrentKernelIndex,
                NumberOfKernelArgs,
                true,
                Result...>
            {
                using type = std::tuple<Result...>;
            };

            // Entry point
            template<typename TupleArgs, typename Indices, typename ReplacementTuple>
            struct ReplaceAtIndices
            {
                static constexpr std::size_t N = std::tuple_size_v<TupleArgs>;
                static_assert(N != 0, "N value (force print)");
                using type = typename rebuild_tuple_impl<TupleArgs, Indices, ReplacementTuple, 0, N, (0 >= N)>::type;
                // static_assert(std::is_same_v<type, void()>);
            };
            template<typename Indices, typename KernelArgsTuple, typename CombinationTuple>
            struct KernelVersions;

            template<typename Indices, typename KernelArgsTuple, typename... Combinations>
            struct KernelVersions<Indices, KernelArgsTuple, std::tuple<Combinations...>>
            {
                using type = std::tuple<typename ReplaceAtIndices<KernelArgsTuple, Indices, Combinations>::type...>;
            };

            template<template<typename...> class Template, typename Tuple>
            struct ApplyTupleToTemplate;

            template<template<typename...> class Template, typename... Args>
            struct ApplyTupleToTemplate<Template, std::tuple<Args...>>
            {
                using type = Template<Args...>;
            };
            template<template<typename...> class Kernel, typename TupleOfTuples>
            struct InstantiateKernelsFromTuple;

            template<template<typename...> class Kernel, typename... ArgTuples>
            struct InstantiateKernelsFromTuple<Kernel, std::tuple<ArgTuples...>>
            {
                using type = std::tuple<typename ApplyTupleToTemplate<Kernel, ArgTuples>::type...>;
            };
        } // namespace createNewKernel

        template<typename>
        struct GetTemplate;

        template<template<typename...> class Template, typename... Args>
        struct GetTemplate<Template<Args...>>
        {
            template<typename... Ts>
            using type = Template<Ts...>;
        };
        template<template<typename...> class Template, typename Tuple>
        struct ApplyTupleToTemplate;

        template<template<typename...> class Template, typename... Args>
        struct ApplyTupleToTemplate<Template, std::tuple<Args...>>
        {
            using type = Template<Args...>;
        };
        template<typename KernelInstance, typename TupleOfTuples>
        struct InstantiateKernelsFromTuple;

        template<typename KernelInstance, typename... ArgTuples>
        struct InstantiateKernelsFromTuple<KernelInstance, std::tuple<ArgTuples...>>
        {
        private:
            // Extract the underlying class template
            template<typename... Ts>
            using Template = typename GetTemplate<KernelInstance>::template type<Ts...>;

        public:
            using type = std::tuple<typename ApplyTupleToTemplate<Template, ArgTuples>::type...>;
        };

        template<typename KernelTuple, std::size_t... Is>
        constexpr auto getValues(std::index_sequence<Is...>)
        {
            KernelTuple initiatedTuple{};
            static_assert(
                (::alpaka::isCVector_v<std::remove_cvref_t<decltype(std::get<Is>(initiatedTuple))>> && ...),
                "The (with tuned_indicies) selected Compile Time Tuneable Arguments of your Kernel type must all be "
                "CVectors");
            return std::make_tuple(std::get<Is>(initiatedTuple)...);
        }

        template<typename T, uint32_t N, T... Vals>
        constexpr auto toRuntimeVec(Vec<T, N, ::alpaka::detail::CVec<T, Vals...>> const&)
        {
            return Vec<T, N, ArrayStorage<T, N>>{Vals...};
        }

        template<typename Tuple>
        constexpr auto convertTupleOfCVecsToRuntimeVecs(Tuple const& t)
        {
            return std::apply([](auto const&... cvecs) { return std::make_tuple(toRuntimeVec(cvecs)...); }, t);
        }

        template<typename TupleOfTuples>
        constexpr auto convertAllCVecCombinationsToRuntimeVecs(TupleOfTuples const& all)
        {
            return std::apply(
                [](auto const&... cvecTuple)
                { return std::make_tuple(convertTupleOfCVecsToRuntimeVecs(cvecTuple)...); },
                all);
        }
    } // namespace CompileTimeHelpers

    namespace trait
    {
        template<typename KernelFn>
        struct RegisteredCTuneables
        {
            using FromTrait = alpaka::tune::trait::CompileTimeTuneableTrait<std::decay_t<KernelFn>>;
            using TuneDefsTuple = decltype(FromTrait::tuneAbleDefinitions());
            static constexpr auto tuneAbleDefinitions = FromTrait::tuneAbleDefinitions();
            static constexpr std::size_t numDefs = std::tuple_size_v<TuneDefsTuple>;
            static constexpr std::size_t numIndices = getDim(FromTrait::tuned_indices);
            static_assert(
                numDefs == numIndices,
                "Mismatch: number of tuneable definitions must match dimension of tuned_indices");

            using ExpandedTuples
                = decltype(alpaka::tune::CompileTimeHelpers::expandFromDefinition::expandValuesFromDefinition<
                           decltype(FromTrait::tuneAbleDefinitions())>());

            using AllCombinations =
                typename alpaka::tune::CompileTimeHelpers::allCombinations::CartesianFromTuple<ExpandedTuples>::type;

            static constexpr auto rCombinations = convertAllCVecCombinationsToRuntimeVecs(AllCombinations{});

            using KernelTuple =
                typename alpaka::tune::CompileTimeHelpers::createNewKernel::ExtractTemplateArgsFromGenericKernel<
                    KernelFn>::type;

            using T_KernelArguments = typename alpaka::tune::CompileTimeHelpers::createNewKernel::
                KernelVersions<decltype(FromTrait::tuned_indices), KernelTuple, AllCombinations>::type;
            using T_KernelVariants = typename alpaka::tune::CompileTimeHelpers::
                InstantiateKernelsFromTuple<KernelFn, T_KernelArguments>::type;
            static constexpr auto KernelVariants = T_KernelVariants{};
            static constexpr auto KernelInitialValues
                = getValues<KernelTuple>(toIntegerSequence(FromTrait::tuned_indices));
        };

        template<typename KernelFn>
        constexpr auto registeredCTuneables()
        {
            return RegisteredCTuneables<KernelFn>{};
        }
    } // namespace trait

    namespace CompileTimeHelpers
    {
        template<typename Tuple, typename Fn, std::size_t... Is>
        void runtime_tuple_dispatch_impl(std::size_t i, Tuple& tup, Fn&& fn, std::index_sequence<Is...>)
        {
            // Use a fold expression to emulate a switch
            bool matched = ((i == Is ? (fn(std::get<Is>(tup)), true) : false) || ...);
            if(!matched)
                throw std::out_of_range("Index out of range");
        }

        template<typename MapType, typename KernelFn>
        auto constructMap()
        {
            auto map = MapType{};
            std::apply(
                [&](auto&&... entries)
                {
                    std::size_t index = 0;
                    ((map[entries] = index++), ...);
                },
                trait::RegisteredCTuneables<KernelFn>::rCombinations // no `typename` here
            );

            return map;
        }
    } // namespace CompileTimeHelpers

    namespace trait
    {
        template<typename KernelFn, typename... Args>
        auto constructRuntimeCtuneablesForActivKernel(alpaka::KernelBundle<KernelFn, Args...>& kernelBundle)
        {
            if constexpr(!hasUserDefinedCTuneable<KernelFn>::value)
            {
                return std::tuple<>{};
            }
            else
            {
                constexpr auto ctune = trait::registeredCTuneables<KernelFn>();
                constexpr auto& definitions = decltype(ctune)::tuneAbleDefinitions;
                constexpr auto& KernelInitialValues = decltype(ctune)::KernelInitialValues;

                constexpr std::size_t N = std::tuple_size_v<std::decay_t<decltype(definitions)>>;
                // use index sequence to access each tuple element
                return [&]<std::size_t... Is>(std::index_sequence<Is...>)
                {
                    return std::make_tuple(
                        ::alpaka::tune::Tuneable<
                            decltype(toRuntimeVec(
                                typename std::decay_t<decltype(std::get<Is>(definitions))>::T_Begin{})),
                            std::decay_t<decltype(std::get<Is>(definitions))>::tag,
                            ::alpaka::tune::DimensionsDependent>{
                            toRuntimeVec(std::get<Is>(KernelInitialValues)),
                            ::alpaka::IdxRange{
                                toRuntimeVec(typename std::decay_t<decltype(std::get<Is>(definitions))>::T_Begin{}),
                                toRuntimeVec(typename std::decay_t<decltype(std::get<Is>(definitions))>::T_End{}),
                                toRuntimeVec(typename std::decay_t<decltype(std::get<Is>(definitions))>::T_Stride{})},
                            "CTune_" + std::to_string(Is)}...);
                }(std::make_index_sequence<N>{});
            }
        }

        template<typename KernelFn, typename... Args>
        static auto& getRtimeIndexMap(alpaka::KernelBundle<KernelFn, Args...> const& kernelBundle)
        {
            using KeyType = std::tuple_element_t<0, decltype(trait::RegisteredCTuneables<KernelFn>::rCombinations)>;
            using MapType = std::unordered_map<KeyType, std::size_t, TuneableTupleHash>;
            static MapType map = CompileTimeHelpers::constructMap<MapType, KernelFn>();
            return map; // should return a non-owning reference to the pointer that resides in that scope..
        };
    } // namespace trait

    template<typename Tuple, typename Fn>
    void runtime_Kernel_dispatch(std::size_t i, Tuple& tup, Fn&& fn)
    {
        constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
        CompileTimeHelpers::runtime_tuple_dispatch_impl(i, tup, std::forward<Fn>(fn), std::make_index_sequence<N>{});
    }
} // namespace alpaka::tune
#endif // COMPILETIMETEMPLATES_H
