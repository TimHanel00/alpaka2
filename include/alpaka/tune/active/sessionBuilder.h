
#ifndef SESSIONBUILDER_HPP
#define SESSIONBUILDER_HPP
#include "alpaka/onHost.hpp"
#include "alpaka/tune/active/tuningEnvironment.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/activeKernel.hpp>
#include <alpaka/tune/active/constraint.hpp>
#include <alpaka/tune/active/strategy.hpp>
#include <alpaka/tune/utils/TimeEvent.hpp>
#include <alpaka/tune/utils/tupleHandle.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace alpaka
{
    template<typename T_Strategy, typename T_Constraint, typename... T_KernelRunArgs>
    struct TuningSession;
} // namespace alpaka

namespace alpaka::tune
{


    template<
        typename T_Strategy = strategy::randomSearch<Timing>,
        typename T_ConstraintTuple = std::tuple<>,
        typename... T_KernelRunArgs>
    class TuningBuilder
    {
    public:
        TuningBuilder() = default;
        T_ConstraintTuple m_constraintTuple;

        explicit TuningBuilder(T_ConstraintTuple constraints, ActiveKernelRun<T_KernelRunArgs...> const& run)
            : m_constraintTuple(constraints)
            , m_run(run)
        {
        }

        template<auto N, typename T>
        auto withTuning(Tuneable<N, T> tuningObject) const
        {
            std::cout << " before with Tuning, " << m_run.toHash() << std::endl;
            auto newRun = appendTuning(m_run, tuningObject);
            std::cout << " after with Tuning, " << newRun.toHash() << std::endl;
            auto ret = helperCreateNewBuilder<T_Strategy, T_ConstraintTuple>(m_constraintTuple, newRun);
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            return ret;
        }

        template<auto... Names, typename T_Predicate>
        auto withConstraint(T_Predicate pred)
        {
            auto constraint = Constraint<T_Predicate, Names...>{pred};
            auto newTuple = std::tuple_cat(m_constraintTuple, std::make_tuple(constraint));

            using T_ConstraintTupleNew = decltype(newTuple);
            auto ret = helperCreateNewBuilder<T_Strategy, T_ConstraintTupleNew>(newTuple, m_run);
            return ret;
        }

        // -- 2. Split the tuple into strings and lambda
        template<typename Tuple>
        auto withConstraintDispatch(Tuple&& tuple) const
        {
            constexpr std::size_t N = std::tuple_size<std::decay_t<Tuple>>::value;
            static_assert(
                N >= 2,
                "withConstraint requires at least one name and a predicate lambda. Example: .withConstraint(\" "
                "tuneableName A \",\" "
                "tuneableName B\",[](alpaka::concepts::Vector auto a,alpaka::concepts::Vector auto b){return "
                "a[0]<b[0];}");

            constexpr std::size_t Last = N - 1;
            return withConstraintDispatchImpl(
                std::forward<Tuple>(tuple),
                std::get<Last>(std::forward<Tuple>(tuple)),
                std::make_index_sequence<Last>{});
        }

        // -- 3. Use std::apply to correctly pack args
        template<typename Tuple, typename Predicate, std::size_t... Is>
        auto withConstraintDispatchImpl(Tuple&& tuple, Predicate&& lambda, std::index_sequence<Is...>) const
        {
            auto namesTuple = std::make_tuple(StaticString<sizeof(std::get<Is>(tuple))>{std::get<Is>(tuple)}...);
            return withConstraintImpl(namesTuple, std::forward<Predicate>(lambda));
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
            TuningBuilder<NewStrategy, T_KernelRunArgs...> ret;
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

        auto withNumBlocksTune() const
        {
            auto tuningObject = makeNumBlocksTune();
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withNumBlocksTune(Tuneable<gridSizeName, T> tune) const
        {
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withNumBlocksTune(alpaka::Vec<T, dim> tune) const
        {
            auto tuningObject = makeNumBlocksTune(tune);
            return this->withTuning(tuningObject);
        }

        auto withBlockSizeTune() const
        {
            auto tuningObject = makeThreadBlockSizeTune();
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withBlockSizeTune(Tuneable<threadBlockSizeName, T> tune) const
        {
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune) const
        {
            auto tuningObject = makeThreadBlockSizeTune(tune);
            return this->withTuning(tuningObject);
        }

        auto withNumFramesTune() const
        {
            auto tuningObject = makeNumFramesTune();
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withNumFramesTune(Tuneable<numFramesName, T> tune) const
        {
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withNumFramesTune(alpaka::Vec<T, dim> tune) const
        {
            auto tuningObject = makeNumFramesTune(tune);
            return this->withTuning(tuningObject);
        }

        auto withFrameExtentTune() const
        {
            auto tuningObject = makeFrameExtentTune();
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withFrameExtentTune(Tuneable<frameExtentName, T> tune) const
        {
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withFrameExtentTune(alpaka::Vec<T, dim> tune) const
        {
            auto tuningObject = makeFrameExtentTune(tune);
            return this->withTuning(tuningObject);
        }

        // Output a fully constructed TuningSession
        auto build() const
        {
            return TuningSession<T_Strategy, T_ConstraintTuple, T_KernelRunArgs...>(
                T_Strategy{},
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

    template<typename T_Strategy_, typename T_ConstraintTuple, typename... Args>
    auto helperCreateNewBuilder(T_ConstraintTuple const& newTuple, ActiveKernelRun<Args...> const& run)
    {
        return TuningBuilder<T_Strategy_, T_ConstraintTuple, Args...>(newTuple, run);
    }
}; // namespace alpaka::tune
#endif
