//
// Created by tim on 06.05.25.
//

#ifndef COMPILETIMETEMPLATES_H

#define COMPILETIMETEMPLATES_H
#include <alpaka/Vec.hpp>
#include <alpaka/tune/traits/traits.hpp>

namespace alpaka::tune
{

    namespace CompileTimeHelpers
    {


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
                using current_T = std::tuple_element_t<CurrentKernelIndex, Tuple>;
                using index_in_T = index_in<CurrentKernelIndex, std::remove_cvref_t<Indices>>;

                static constexpr std::size_t indexWhereCurrentKernelIndexWasFound
                    = (index_in_T::index == static_cast<std::size_t>(-1)) ? 0 : index_in_T::index;

                using replacement = std::tuple_element_t<indexWhereCurrentKernelIndexWasFound, Replacements>;
                // Check type and dimension if both are CVec
                // static constexpr bool typeMatches = CVectorCompatible<replacement, current_T>::value;


                // static constexpr bool shouldReplace = index_in_T::value && typeMatches;
                static constexpr bool shouldReplace = index_in_T::value;
                static constexpr bool isDone = (CurrentKernelIndex + 1 >= NumberOfKernelArgs);

                using type = typename std::conditional_t<
                    shouldReplace,
                    rebuild_tuple_impl<
                        Tuple,
                        Indices,
                        Replacements,
                        CurrentKernelIndex + 1,
                        NumberOfKernelArgs,
                        isDone,
                        Result...,
                        replacement>,
                    rebuild_tuple_impl<
                        Tuple,
                        Indices,
                        Replacements,
                        CurrentKernelIndex + 1,
                        NumberOfKernelArgs,
                        isDone,
                        Result...,
                        current_T>>::type;
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
        template<typename TupleOfTuples>
        struct TupleSizeSequence;

        template<typename... Tuples>
        struct TupleSizeSequence<std::tuple<Tuples...>>
        {
            using type = std::index_sequence<std::tuple_size_v<Tuples>...>;
        };

        template<typename T, T... Ns>
        constexpr auto to_array(std::integer_sequence<T, Ns...>)
        {
            return std::array<T, sizeof...(Ns)>{Ns...};
        }

        template<typename TupleOfTuples>
        using TupleSizeSequence_t = typename TupleSizeSequence<TupleOfTuples>::type;
        template<typename T>
        struct unwrapCTunablesToTuples;

        // Specialization: for a tuple of CTunables
        template<typename... CTs>
        struct unwrapCTunablesToTuples<std::tuple<CTs...>>
        {
            using type = std::tuple<typename CTs::Values...>;
        };

        // Helper alias
        template<typename T>
        using unwrapCTunablesToTuples_t = typename unwrapCTunablesToTuples<T>::type;


    } // namespace CompileTimeHelpers

    namespace trait
    {
        template<typename KernelFn>
        struct RegisteredCTuneables
        {
            using FromTrait = alpaka::tune::trait::CompileTimeTuneableTrait<std::decay_t<KernelFn>>;
            using TuneDefsTuple = decltype(FromTrait::tuneAbleDefinitions());
            static constexpr std::size_t numDefs = std::tuple_size_v<TuneDefsTuple>;
            static constexpr std::size_t numIndices = getDim(FromTrait::tuned_indices);
            static_assert(
                numDefs == numIndices,
                "Mismatch: number of tuneable definitions must match dimension of tuned_indices");


            using unwrappedCTuneableTuples =
                typename alpaka::tune::CompileTimeHelpers::unwrapCTunablesToTuples<TuneDefsTuple>::type;
            using Sizes = CompileTimeHelpers::TupleSizeSequence_t<unwrappedCTuneableTuples>;
            //  GEk<Flattened> expanded;
            using AllCombinations = typename alpaka::tune::CompileTimeHelpers::allCombinations::CartesianFromTuple<
                unwrappedCTuneableTuples>::type;

            using KernelTuple =
                typename alpaka::tune::CompileTimeHelpers::createNewKernel::ExtractTemplateArgsFromGenericKernel<
                    KernelFn>::type;

            using T_KernelArguments = typename alpaka::tune::CompileTimeHelpers::createNewKernel::
                KernelVersions<decltype(FromTrait::tuned_indices), KernelTuple, AllCombinations>::type;
            using T_KernelVariants = typename alpaka::tune::CompileTimeHelpers::
                InstantiateKernelsFromTuple<KernelFn, T_KernelArguments>::type;
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

        template<typename KernelFn, auto Dim, concepts::Integral Integral>
        inline Integral calculateRowMajorIndex(std::array<Integral, Dim> const& indicies)
        {
            using sizes_T = typename trait::RegisteredCTuneables<KernelFn>::Sizes;
            constexpr auto sizes = to_array(sizes_T{});
            Integral idx = 0;
            static_assert(
                Dim == sizes_T::size(),
                "Input Index for Kernel Variant Access does not match expected sizes!");
            for(auto i = 0u; i < Dim; ++i)
            {
                idx = idx * sizes[i] + indicies[i];
            }
            return idx;
        }

        template<typename KernelFn, auto Dim, concepts::Integral Integral, typename Fn>
        void runtime_Kernel_dispatch(std::array<Integral, Dim> const& indicies, Fn&& fn)
        {
            static constexpr auto variants =
                typename trait::RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants{};
            static constexpr auto numVars
                = std::tuple_size_v<typename trait::RegisteredCTuneables<std::decay_t<KernelFn>>::T_KernelVariants>;
            uint32_t rowMajorIndex = CompileTimeHelpers::calculateRowMajorIndex<KernelFn>(indicies);
            CompileTimeHelpers::runtime_tuple_dispatch_impl(
                rowMajorIndex,
                variants,
                std::forward<Fn>(fn),
                std::make_index_sequence<numVars>{});
        }
    } // namespace CompileTimeHelpers


} // namespace alpaka::tune
#endif // COMPILETIMETEMPLATES_H
