
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


    template<
        typename T_Strategy = strategy::randomSearch,
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
            ActiveKernelRun<T_KernelRunArgs...> const& run)
            : m_constraintTuple(constraints)
            , m_run(run)
            , m_strategy(strategy)
            , m_metricInterface(interface)

        {
        }

        template<typename T_objct>
        auto withTuning(T_objct tuningObject) const
        {
            std::cout << " before with Tuning, " << m_run.toHash() << std::endl;
            auto newRun = appendTuning(m_run, tuningObject);
            std::cout << " after with Tuning, " << newRun.toHash() << std::endl;
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
            auto newTune = Tuneable<T, static_cast<std::size_t>(SpecialTuneableID::NumBlocks), Policy>(
                tune.value,
                tune.idxRange,
                tune.m_name);
            return this->withTuning(newTune);
        }

        template<typename T, auto dim, typename DimTraversePolicy = DimensionsIndependent>
        auto withNumBlocksTune(alpaka::Vec<T, dim> tune, DimTraversePolicy = {}) const
        {
            auto newTune = Tuneable<
                alpaka::Vec<T, dim>,
                static_cast<std::size_t>(SpecialTuneableID::NumBlocks),
                DimTraversePolicy>(tune);
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
            auto newTune = Tuneable<T, static_cast<std::size_t>(SpecialTuneableID::ThreadBlock), Policy>(
                tune.value,
                tune.idxRange,
                tune.m_name);
            return this->withTuning(newTune);
        }

        template<typename T, auto dim, typename DimTraversePolicy = DimensionsIndependent>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune, DimTraversePolicy = {}) const
        {
            auto newTune = Tuneable<
                alpaka::Vec<T, dim>,
                static_cast<std::size_t>(SpecialTuneableID::ThreadBlock),
                DimTraversePolicy>(tune);
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
            auto newTune = Tuneable<T, static_cast<std::size_t>(SpecialTuneableID::NumFrames), Policy>(
                tune.value,
                tune.idxRange,
                tune.m_name);
            return this->withTuning(newTune);
        }

        template<typename T, auto dim, typename DimTraversePolicy = DimensionsIndependent>
        auto withNumFramesTune(alpaka::Vec<T, dim> tune, DimTraversePolicy = {}) const
        {
            auto newTune = Tuneable<
                alpaka::Vec<T, dim>,
                static_cast<std::size_t>(SpecialTuneableID::NumFrames),
                DimTraversePolicy>(tune);
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
            auto newTune = Tuneable<T, static_cast<std::size_t>(SpecialTuneableID::FrameExtent), Policy>(
                tune.value,
                tune.idxRange,
                tune.m_name);
            return this->withTuning(newTune);
        }

        template<typename T, auto dim, typename DimTraversePolicy = DimensionsIndependent>
        auto withFrameExtentTune(alpaka::Vec<T, dim> tune, DimTraversePolicy = {}) const
        {
            auto newTune = Tuneable<
                alpaka::Vec<T, dim>,
                static_cast<std::size_t>(SpecialTuneableID::FrameExtent),
                DimTraversePolicy>(tune);
            return this->withTuning(newTune);
        }

        // Output a fully constructed TuningSession
        auto build() const
        {
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

        ActiveKernelRun<T_KernelRunArgs...> m_run;
    };

    template<typename T_Strategy_, typename T_MetricInterface, typename T_ConstraintTuple, typename... Args>
    auto helperCreateNewBuilder(
        T_Strategy_ const& strategy,
        T_MetricInterface const& interface,
        T_ConstraintTuple const& newTuple,
        ActiveKernelRun<Args...> const& run)
    {
        return TuningBuilder<T_Strategy_, T_MetricInterface, T_ConstraintTuple, Args...>(
            strategy,
            interface,
            newTuple,
            run);
    }
}; // namespace alpaka::tune
#endif
