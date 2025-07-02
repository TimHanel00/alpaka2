
#ifndef SESSIONBUILDER_HPP
#define SESSIONBUILDER_HPP
#include "alpaka/onHost.hpp"
#include "alpaka/tune/active/tuningEnvironment.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/MetricInterface.hpp>
#include <alpaka/tune/active/constraint.hpp>
#include <alpaka/tune/active/strategy.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace alpaka
{
    template<typename T_Strategy, typename T_MetricInterface, typename T_Constraint, typename... T_KernelRunArgs>
    struct TuningSession;
} // namespace alpaka

namespace alpaka::tune
{
    template<typename T_Strategy, typename T_MetricInterface, typename T_ConstraintTuple, typename... T_KernelRunArgs>
    class TuningBuilder;

    template<typename T_Strategy_, typename T_MetricInterface, typename T_ConstraintTuple, typename... Args>
    auto helperCreateNewBuilder(
        T_Strategy_ const& strategy,
        T_MetricInterface const& interface,
        T_ConstraintTuple const& newTuple,
        KernelTuningModel<Args...> const& run)
    {
        return TuningBuilder<T_Strategy_, T_MetricInterface, T_ConstraintTuple, Args...>(
            strategy,
            interface,
            newTuple,
            run);
    }

    template<auto ID, typename T, auto ID2, typename DimIndep>
    auto helperChangeTuneableID(Tuneable<T, ID2, DimIndep> const& tune)
    {
        std::cout << " val " << tune.value.toString() << std::endl;
        auto createTune = Tuneable<T, ID, DimIndep>(tune.idxRange, tune.value);
        createTune.inputList = tune.inputList;
        createTune.hasRange = tune.hasRange;
        std::cout << " create " << createTune.value.toString() << std::endl;
        return createTune;
    };

    namespace strategy::detail
    {
#ifdef strategy_randomSearch
        inline std::string strat_name = "randomSearch";
#elif strategy_exhaustiveSearch
        inline std::string strat_name = "exhaustiveSearch";
#elif strategy_simulatedAnnealing
        inline std::string strat_name = "simulatedAnnealing";
#elif strategy_randomSample
        inline std::string strat_name = "randomSample";
#else
        inline std::string strat_name = "randomSearch";
#endif
        static auto getName()
        {
            return strat_name;
        }
    } // namespace strategy::detail
    template<

#ifdef strategy_randomSearch
        typename T_Strategy = alpaka::tune::strategy::randomSearch,
#elif strategy_exhaustiveSearch
        typename T_Strategy = alpaka::tune::strategy::exhaustiveSearch,
#elif strategy_simulatedAnnealing
        typename T_Strategy = alpaka::tune::strategy::simulatedAnnealing,
#elif strategy_randomSample
            typename T_Strategy=alpaka::tune::strategy::randomSample
#else
        typename T_Strategy = alpaka::tune::strategy::exhaustiveSearch,
#endif
        typename T_MetricInterface = alpaka::tune::metricInterface::Timing,
        typename T_ConstraintTuple = std::tuple<>,
        typename... T_KernelRunArgs>
    class TuningBuilder
    {
    public:
        TuningBuilder() = default;
        T_ConstraintTuple m_constraintTuple;
        T_Strategy m_strategy{};
        T_MetricInterface m_metricInterface{};

        explicit TuningBuilder(
            T_Strategy strategy,
            T_MetricInterface interface,
            T_ConstraintTuple constraints,
            KernelTuningModel<T_KernelRunArgs...> const& run)
            : m_constraintTuple(constraints)
            , m_run(run)
            , m_strategy(strategy)
            , m_metricInterface(interface)

        {
        }

        template<typename T_objct>
        auto withTuning(T_objct tuningObject) const
        {
            auto newRun = appendTuning(m_run, tuningObject);
            auto ret = helperCreateNewBuilder<T_Strategy, T_MetricInterface, T_ConstraintTuple>(
                m_strategy,
                m_metricInterface,
                m_constraintTuple,
                newRun);
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            return ret;
        }

        template<auto... TuneableIDs, typename T_Predicate>
        auto constraintHelper(T_Predicate pred) const
        {
            return Constraint<T_Predicate, TuneableIDs...>{pred};
        }

        template<auto... TuneableIDs, typename T_Predicate>
        auto withConstraint(T_Predicate pred)
        {
            auto constraint = Constraint<T_Predicate, TuneableIDs...>{pred};
            auto newTuple = std::tuple_cat(m_constraintTuple, std::make_tuple(constraint));

            using T_ConstraintTupleNew = decltype(newTuple);
            auto ret = helperCreateNewBuilder<T_Strategy, T_MetricInterface, T_ConstraintTupleNew, T_KernelRunArgs...>(
                m_strategy,
                m_metricInterface,
                newTuple,
                m_run);
            return ret;
        }

        TuningBuilder& withReRuns(std::size_t reruns)
        {
            m_reRuns = reruns;
            return *this;
        }

        TuningBuilder& withConfig(std::string config)
        {
            m_config = std::move(config);
            return *this;
        }

        TuningBuilder& withDynamicRuns(std::size_t runs)
        {
            m_dynamicRuns = runs;
            return *this;
        }

        template<typename NewStrategy>
        auto withStrategy(NewStrategy strategy) const
        {
            auto ret = helperCreateNewBuilder<NewStrategy, T_MetricInterface, T_ConstraintTuple, T_KernelRunArgs...>(
                strategy,
                m_metricInterface,
                m_constraintTuple,
                m_run);
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_run = m_run;
            return ret;
        }

        template<typename... T_Specifiers>
        TuningBuilder& withRunSpecifiers(T_Specifiers... specifiers)
        {
            processArgs(m_sessionSpecifiers, specifiers...);
            return *this;
        }

        template<typename DimTraversePolicy = DimensionsIndependent>
        auto withNumBlocksTune(DimTraversePolicy = {}) const
        {
            auto tuningObject = Tuneable<
                alpaka::Vec<std::size_t, 1>,
                static_cast<std::size_t>(SpecialTuneableID::NumBlocks),
                DimTraversePolicy>{};
            return this->withTuning(tuningObject);
        }

        template<typename T, auto ID, typename Policy>
        auto withNumBlocksTune(Tuneable<T, ID, Policy> tune) const
        {
            auto newTune = helperChangeTuneableID<static_cast<std::size_t>(SpecialTuneableID::NumBlocks)>(tune);
            return this->withTuning(newTune);
        }

        //=============================
        // BlockSize
        //=============================

        template<typename DimTraversePolicy = DimensionsIndependent>
        auto withBlockSizeTune(DimTraversePolicy = {}) const
        {
            auto tuningObject = Tuneable<
                alpaka::Vec<std::size_t, 1>,
                static_cast<std::size_t>(SpecialTuneableID::ThreadBlock),
                DimTraversePolicy>{};
            return this->withTuning(tuningObject);
        }

        template<typename T, auto ID, typename Policy>
        auto withBlockSizeTune(Tuneable<T, ID, Policy> tune) const
        {
            auto newTune = helperChangeTuneableID<static_cast<std::size_t>(SpecialTuneableID::ThreadBlock)>(tune);
            return this->withTuning(newTune);
        }

        //=============================
        // NumFrames
        //=============================

        template<typename DimTraversePolicy = DimensionsIndependent>
        auto withNumFramesTune(DimTraversePolicy = {}) const
        {
            auto tuningObject = Tuneable<
                alpaka::Vec<std::size_t, 1>,
                static_cast<std::size_t>(SpecialTuneableID::NumFrames),
                DimTraversePolicy>{};
            return this->withTuning(tuningObject);
        }

        template<typename T, auto ID, typename Policy>
        auto withNumFramesTune(Tuneable<T, ID, Policy> tune) const
        {
            auto newTune = helperChangeTuneableID<static_cast<std::size_t>(SpecialTuneableID::NumFrames)>(tune);
            return this->withTuning(newTune);
        }

        //=============================
        // FrameExtent
        //=============================

        template<typename DimTraversePolicy = DimensionsIndependent>
        auto withFrameExtentTune(DimTraversePolicy = {}) const
        {
            auto tuningObject = Tuneable<
                alpaka::Vec<std::size_t, 1>,
                static_cast<std::size_t>(SpecialTuneableID::FrameExtent),
                DimTraversePolicy>{};
            return this->withTuning(tuningObject);
        }

        template<typename T, auto ID, typename Policy>
        auto withFrameExtentTune(Tuneable<T, ID, Policy> tune) const
        {
            auto newTune = helperChangeTuneableID<static_cast<std::size_t>(SpecialTuneableID::FrameExtent)>(tune);
            return this->withTuning(newTune);
        }

        // Output a fully constructed TuningSession
        auto build() const
        {
            using run_BareT = std::remove_cvref_t<decltype(m_run)>;
            constexpr auto condA = run_BareT::hasNumBlocksTune() && run_BareT::hasNumFramesTune();
            constexpr auto condB = run_BareT::hasThreadBlockSizeTune() && run_BareT::hasFrameExtentTune();
            if constexpr(condA || condB)
            {
                if constexpr(condA && condB)
                {
                    auto framesSmallerBlocksCondition
                        = this->template constraintHelper<frameTune::NumFrames, frameTune::numBlocks>(
                            []<typename T0>(T0 a, auto b)
                            {
                                using T = T0;
                                bool allTrue = true;
                                for(int i = 0; i < T::dim(); i++)
                                    allTrue = allTrue && (a[i] >= b[i]);
                                return allTrue;
                            });
                    auto threadsSmallerExtentCondition
                        = this->template constraintHelper<frameTune::FrameExtent, frameTune::ThreadBlock>(
                            []<typename T0>(T0 a, auto b)
                            {
                                using T = T0;
                                bool allTrue = true;
                                for(int i = 0; i < T::dim(); i++)
                                    allTrue = allTrue && (a[i] >= b[i]);
                                return allTrue;
                            });

                    auto newTuple = std::tuple_cat(
                        m_constraintTuple,
                        std::make_tuple(threadsSmallerExtentCondition),
                        std::make_tuple(framesSmallerBlocksCondition));
                    return TuningSession<T_Strategy, T_MetricInterface, decltype(newTuple), T_KernelRunArgs...>(
                        m_strategy,
                        m_metricInterface,
                        newTuple,
                        m_config,
                        m_reRuns.value_or(0),
                        m_dynamicRuns.value_or(0),
                        m_sessionSpecifiers,
                        m_run);
                }
                else if constexpr(condA)
                {
                    auto framesSmallerBlocksCondition
                        = this->template constraintHelper<frameTune::NumFrames, frameTune::numBlocks>(
                            []<typename T0>(T0 a, auto b)
                            {
                                using T = T0;
                                bool allTrue = true;
                                for(int i = 0; i < T::dim(); i++)
                                    allTrue = allTrue && (a[i] >= b[i]);
                                return allTrue;
                            });
                    auto newTuple = std::tuple_cat(m_constraintTuple, std::make_tuple(framesSmallerBlocksCondition));
                    return TuningSession<T_Strategy, T_MetricInterface, decltype(newTuple), T_KernelRunArgs...>(
                        m_strategy,
                        m_metricInterface,
                        newTuple,
                        m_config,
                        m_reRuns.value_or(0),
                        m_dynamicRuns.value_or(0),
                        m_sessionSpecifiers,
                        m_run);
                }
                else
                {
                    std::cout << " check frameExtent bigger condition" << std::endl;
                    auto threadsSmallerExtentCondition
                        = this->template constraintHelper<frameTune::FrameExtent, frameTune::ThreadBlock>(
                            []<typename T0>(T0 a, auto b)
                            {
                                using T = T0;
                                bool allTrue = true;
                                for(int i = 0; i < T::dim(); i++)
                                    allTrue = allTrue && (a[i] >= b[i]);
                                return allTrue;
                            });
                    auto newTuple = std::tuple_cat(m_constraintTuple, std::make_tuple(threadsSmallerExtentCondition));
                    return TuningSession<T_Strategy, T_MetricInterface, decltype(newTuple), T_KernelRunArgs...>(
                        m_strategy,
                        m_metricInterface,
                        newTuple,
                        m_config,
                        m_reRuns.value_or(0),
                        m_dynamicRuns.value_or(0),
                        m_sessionSpecifiers,
                        m_run);
                }
            }
            else

                return TuningSession<T_Strategy, T_MetricInterface, T_ConstraintTuple, T_KernelRunArgs...>(
                    m_strategy,
                    m_metricInterface,
                    m_constraintTuple,
                    m_config,
                    m_reRuns.value_or(0),
                    m_dynamicRuns.value_or(0),
                    m_sessionSpecifiers,
                    m_run);
        }

        std::optional<std::size_t> m_reRuns;
        std::optional<std::size_t> m_dynamicRuns;
        std::string m_config;
        std::vector<std::string> m_sessionSpecifiers;

        KernelTuningModel<T_KernelRunArgs...> m_run;

    private:
        // numframe,frameExtent,numBlocks,numThreads
        bool frameTuneExists[4] = {false, false, false, false};
    };


}; // namespace alpaka::tune
#endif
