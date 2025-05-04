//
// Created by tim on 01.05.25.
//

#ifndef TRAITS_HPP
#define TRAITS_HPP
#include "../../../../example/heatEquation2D/src/StencilKernel.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>

struct StencilKernel2;

namespace alpaka::tune::trait
{
    struct postProcessing
    {
        template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename KernelFn, typename... args>
        struct Op
        {
            void operator()(
                T_Config& config,
                T_FrameSpec& frame_spec,
                T_Metric& metricInterface,
                alpaka::KernelBundle<KernelFn, args...> const& kernel)
            {
            }
        };
    };

    template<typename T_KernelBundle, typename T_Config, typename T_FrameSpec, typename T_Metric>
    inline auto callPostProcessing(

        T_Config& config,
        T_FrameSpec& frame_spec,
        T_Metric& metricInterface,
        T_KernelBundle& KernelBundle)
    {
        return postProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(

            config,
            frame_spec,
            metricInterface,
            KernelBundle);
    }

    // default
    struct preProcessing
    {
        template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename KernelFn, typename... args>
        struct Op
        {
            void operator()(
                T_Config& config,
                T_FrameSpec& frame_spec,
                T_Metric& metricInterface,
                alpaka::KernelBundle<KernelFn, args...> const& kernel)
            {
            }
        };
    };

    /*
    // example specialization for a Kernel HostSideKernel
    template<typename T_Config, typename T_FrameSpec, typename T_Metric, typename... args1, typename... args2>
    struct preProcessing::Op<T_Config, T_FrameSpec, T_Metric, alpaka::KernelBundle<HostSideKernel<args1...>, args2...>>
    {
        void operator()(
            T_Config& config,
            T_FrameSpec& frame_spec,
            T_Metric& metricInterface,
            alpaka::KernelBundle<HostSideKernel<args1...>, args2...> const& kernelBundle)
        {
            std::cout << " special " << std::endl;
        }
    };
    */
    template<typename T_KernelBundle, typename T_Config, typename T_FrameSpec, typename T_Metric>
    auto callPreProcessing(

        T_Config& config,
        T_FrameSpec& frame_spec,
        T_Metric& metricInterface,
        T_KernelBundle const& KernelBundle)
    {
        return preProcessing::Op<T_Config, T_FrameSpec, T_Metric, T_KernelBundle>{}(
            config,
            frame_spec,
            metricInterface,
            KernelBundle);
    }

    namespace alpaka::tune::CompileTimeHelpers
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

            template<
                typename Value, // e.g., CVec<T, ...>
                typename Begin,
                typename End,
                typename Stride,
                std::size_t ID,
                typename DimTraverse>
            struct UnwrappTuneableType<CTunable<Value, Begin, End, Stride, ID, DimTraverse>>
            {
                using T = typename Value::type;
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
                using TemplateType = Template<>;
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
                static constexpr bool typeReturned = index_in_T::value;
                static constexpr std::size_t indexWhereCurrentKernelIndexWasFound
                    = (index_in_T::index == static_cast<std::size_t>(-1)) ? 0 : index_in_T::index;
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

    } // namespace alpaka::tune::CompileTimeHelpers

    template<typename Kernel>
    struct CompileTimeTuneableTrait
    {
        // a Kernel can have several template parameters (some of which may not be tuneable)
        // -- tuned_indices is a CVector that holds a list of positions for tuneables defined in
        // tuneAbleDefinitions()
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            return std::tuple{}; // empty tuple, no tunables
        }
    };

    template<typename... Args>
    struct CompileTimeTuneableTrait<StencilKernel<Args...>>
    {
        static constexpr auto tuned_indices
            = CVec<std::size_t, static_cast<std::size_t>(0), static_cast<std::size_t>(2)>{};

        static constexpr auto tuneAbleDefinitions()
        {
            constexpr auto tune1
                = tune::CTunable<CVec<int, 1, 2>, CVec<int, 0, 0>, CVec<int, 3, 3>, CVec<int, 1, 1>>{};
            constexpr auto tune2
                = tune::CTunable<CVec<int, 3, 3>, CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1, tune2}; // empty tuple, no tunables
        }
    };

    /**
     * the tuner will try to apply the tuple returned by this function to the KernelFn, if the Kernel has
     * non-tuneabel template types<...> see example
     */
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

    template<typename KernelFn, typename... args1>
    auto registerCTuneabels(KernelBundle<KernelFn, args1...>& KernelBundle)

    {
        using fromTrait = CompileTimeTuneableTrait<std::decay_t<KernelFn>>;
        static constexpr auto tuned_indices = fromTrait::tuned_indices;
        static constexpr auto tuneAbleDefinitions = fromTrait::tuneAbleDefinitions();
        using type = KernelFn;
        // static_assert(std::is_same_v<type, void()>);
        //  compare number of tuple elements in tuneAbleDefinitions against number of elements in tuned_indices
        //  if number of tuple elements in tuneAbleDefinitions is zero simply return KernelBundle, if they are
        //  different throw a compiler error (static_assert)
        using TuneDefsTuple = decltype(tuneAbleDefinitions);
        constexpr std::size_t numDefs = std::tuple_size_v<TuneDefsTuple>;
        constexpr std::size_t numIndices = getDim(tuned_indices);
        if constexpr(numDefs == 0)
        {
            return KernelBundle;
        }
        else
        {
            static_assert(
                numDefs == numIndices,
                "Mismatch: number of tuneable definitions must match dimension of tuned_indices");
            // Helper: number of tunable

            using expandedTuples
                = decltype(alpaka::tune::CompileTimeHelpers::expandFromDefinition::expandValuesFromDefinition<
                           decltype(fromTrait::tuneAbleDefinitions())>());
            using allCombinations =
                typename alpaka::tune::CompileTimeHelpers::allCombinations::CartesianFromTuple<expandedTuples>::type;
            using KernelTuple =
                typename alpaka::tune::CompileTimeHelpers::createNewKernel::ExtractTemplateArgsFromGenericKernel<
                    KernelFn>::type;
            // assert that kernelTuple is bigger or equal than TuneDefsTuple
            using KernelVariants = typename alpaka::tune::CompileTimeHelpers::createNewKernel::
                KernelVersions<decltype(tuned_indices), KernelTuple, allCombinations>::type;
            using InstantiatedKernels = typename InstantiateKernelsFromTuple<KernelFn, KernelVariants>::type;
            static_assert(std::is_same_v<InstantiatedKernels, void()>);
            /*
            expandedTuple is a tuple of tuples - containing CVec as types CVec<typename T, T... T_values>
            (list of lists) - each tuple in the big list correspond to a userDefined tuneable definition-

                KernelFn also contains T_Kernel<Args...> or T_Kernel but that should be asserted..

            the task today is to generate a T_Kernel with the corresponding args inserted at tuned_indices of its
            template types for every combination of CVec (where a combination is a tuple selected from expandedTuple
            (selected 1 CVec from every tuple in expandedTuple))*/
            // TODO convert tuneable to KernelFn instantiations. by exhaustive search
        }
    }
} // namespace alpaka::tune::trait
#endif // TRAITS_HPP
