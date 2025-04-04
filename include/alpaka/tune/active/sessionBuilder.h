
#include "alpaka/onHost.hpp"
#include "alpaka/tune/active/KernelSingleton.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/strategy.hpp>
#include <alpaka/tune/utils/TimeEvent.hpp>
#include <alpaka/tune/utils/tupleHandle.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <variant>

namespace alpaka
{
    template<typename T_Strategy, typename T_GridSize, typename T_BlockSize, bool grid, bool block>
    struct TuningSession;
} // namespace alpaka

namespace alpaka::tune
{
    // forward declaration of TuningSession


    template<
        typename T_Strategy = strategy::randomSearch,
        typename T_GridSize = NumBlocksTune<>,
        typename T_BlockSize = ThreadBlockSizeTune<>,
        bool grid = false,
        bool block = false>
    class TuningBuilder
    {
    public:
        TuningBuilder() = default;

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
            TuningBuilder<NewStrategy, T_GridSize, T_BlockSize, grid, block> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = m_gridTune;
            ret.m_blockTune = m_blockTune;
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
            auto tuningObject = NumBlocksTune{};
            tuningObject.userDef = false;
            TuningBuilder<T_Strategy, ALPAKA_TYPEOF(tuningObject), T_BlockSize, true, block> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = std::move(tuningObject);
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withNumBlocksTune(NumBlocksTune<T, T_Begin, T_End, T_Stride> tune) const
        {
            using NewGrid = NumBlocksTune<T, T_Begin, T_End, T_Stride>;
            tune.userDef = true;
            TuningBuilder<T_Strategy, NewGrid, T_BlockSize, true, block> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = std::move(tune);
            return ret;
        }

        template<typename T, auto dim>
        auto withNumBlocksTune(alpaka::Vec<T, dim> tune) const
        {
            using VecType = decltype(tune);
            using NewGrid = NumBlocksTune<VecType, VecType, VecType, VecType>;
            TuningBuilder<T_Strategy, NewGrid, T_BlockSize, true, block> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            auto tuningObject = NewGrid{tune};
            tuningObject.userDef = false;
            ret.m_gridTune = tuningObject;
            return ret;
        }

        auto withBlockSizeTune() const
        {
            auto tuningObject = ThreadBlockSizeTune{};
            tuningObject.userDef = false;
            TuningBuilder<T_Strategy, T_GridSize, ALPAKA_TYPEOF(tuningObject), grid, true> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = m_gridTune;
            ret.m_blockTune = std::move(ThreadBlockSizeTune{});
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withBlockSizeTune(ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride> tune) const
        {
            using NewBlock = ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride>;
            tune.userDef = true;
            TuningBuilder<T_Strategy, T_GridSize, NewBlock, grid, true> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = m_gridTune;
            ret.m_blockTune = std::move(tune);
            return ret;
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune) const
        {
            using VecType = decltype(tune);

            using NewBlock = ThreadBlockSizeTune<VecType, VecType, VecType, VecType>;
            TuningBuilder<T_Strategy, T_GridSize, NewBlock, grid, true> ret;
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
            ret.m_gridTune = m_gridTune;
            auto tuningObject = NewBlock{tune};
            tuningObject.userDef = false;
            ret.m_blockTune = tuningObject;
            return ret;
        }

        // Output a fully constructed TuningSession
        auto build() const
        {
            return TuningSession<T_Strategy, T_GridSize, T_BlockSize, grid, block>(
                T_Strategy{},
                m_config,
                m_reRuns.value_or(0),
                m_dynamicRuns.value_or(0),
                m_sessionSpecifiers,
                m_gridTune,
                m_blockTune);
        }

        std::optional<std::size_t> m_reRuns;
        std::optional<std::size_t> m_dynamicRuns;
        std::string m_config;
        std::vector<std::string> m_sessionSpecifiers;

        T_GridSize m_gridTune;
        T_BlockSize m_blockTune;
    };
}; // namespace alpaka::tune
