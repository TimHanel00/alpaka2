//
// Created by tim on 19.03.25.
//

#ifndef SESSIONBUILDER_H
#define SESSIONBUILDER_H

#include <alpaka/tune/active/tuningSession.hpp>

namespace alpaka::tune
{
    template<
        typename T_Strategy = alpaka::tune::strategy::randomSearch,
        typename T_GridSize = alpaka::tune::GridSizeTune<>,
        typename T_BlockSize = alpaka::tune::ThreadBlockSizeTune<>,
        bool grid = false,
        bool block = false>
    class TuningSessionBuilder
    {
    public:
        using T_floating = double_t;
        using T_Integer = std::size_t;

        ActiveKernelRun<T_GridSize, T_BlockSize, T_floating> run;
        T_Strategy strategy;
        T_Integer dynamicRuns_Nr{0};
        bool m_initialized = false;
        std::size_t reRuns{0};
        std::string config;
        std::vector<std::string> sessionSpecifier;

        // Default constructor
        TuningSessionBuilder() = default;

        explicit TuningSessionBuilder(T_Strategy strategy_) : strategy(strategy_), reRuns(getReRuns())
        {
            run.metric = MetricUndefined;
        }

        // Copy method (same as original)
        template<bool copyGridSize = false, bool copyBlockSize = false, typename T_newBuilder>
        void copy(T_newBuilder& builder)
        {
            builder.config = this->config;
            builder.dynamicRuns_Nr = dynamicRuns_Nr;
            builder.sessionSpecifier = sessionSpecifier;
            builder.reRuns = reRuns;
            builder.run.metric = this->run.metric;
            if constexpr(copyGridSize)
            {
                builder.run.gridSize = this->run.gridSize;
            }
            else if constexpr(copyBlockSize)
            {
                builder.run.threadBlockSize = this->run.threadBlockSize;
            }
        }

        // Set ReRuns
        auto withReRuns(std::size_t const& reRuns_)
        {
            auto ret = *this;
            if(getReRuns() != 0)
            {
                ret.reRuns = reRuns_;
            }
            return ret;
        }

        // Set Config
        auto withConfig(std::string const& config_)
        {
            auto ret = *this;
            ret.config = config_;
            std::cout << " CONFIG: " << config_ << std::endl;
            return ret;
        }

        // Set Run Specifiers
        template<typename... T_Specifiers>
        auto withRunSpecifiers(T_Specifiers... specifiers)
        {
            auto ret = *this;
            processArgs(ret.sessionSpecifier, specifiers...);
            return ret;
        }

        // Grid Size Tune
        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withGridSizeTune(tune::GridSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            using NewGridSize = tune::GridSizeTune<T, T_Begin, T_End, T_Stride>;
            TuningSessionBuilder<T_Strategy, NewGridSize, T_BlockSize, true, block> ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = tune;
            ret.config = this->config;
            return ret;
        }

        template<typename T, auto dim>
        auto withGridSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSessionBuilder<
                T_Strategy,
                tune::GridSizeTune<VecType, VecType, VecType, VecType>,
                T_BlockSize,
                true,
                block>
                ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune<VecType, VecType, VecType, VecType>{tune});
            ret.config = this->config;
            return ret;
        }

        auto withGridSizeTune()
        {
            TuningSessionBuilder<T_Strategy, tune::GridSizeTune<>, T_BlockSize, true, block> ret{strategy};
            this->template copy<false, block>(ret);
            ret.run.gridSize = std::move(tune::GridSizeTune{});
            ret.config = this->config;
            return ret;
        }

        // Block Size Tune
        auto withBlockSizeTune()
        {
            TuningSessionBuilder<T_Strategy, T_GridSize, tune::ThreadBlockSizeTune<>, grid, true> ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune{});
            ret.config = this->config;
            std::cout << " CONFIG: " << ret.config << std::endl;
            return ret;
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune)
        {
            using VecType = ALPAKA_TYPEOF(tune);
            TuningSessionBuilder<
                T_Strategy,
                T_GridSize,
                tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>,
                grid,
                true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = std::move(tune::ThreadBlockSizeTune<VecType, VecType, VecType, VecType>{tune});
            ret.config = this->config;
            std::cout << " CONFIG: " << ret.config << std::endl;
            return ret;
        }

        template<typename T, typename T_Begin, typename T_End, typename T_Stride>
        auto withBlockSizeTune(tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride> tune)
        {
            TuningSessionBuilder<
                T_Strategy,
                T_GridSize,
                tune::ThreadBlockSizeTune<T, T_Begin, T_End, T_Stride>,
                grid,
                true>
                ret{strategy};
            this->template copy<grid, false>(ret);
            ret.run.threadBlockSize = tune;
            ret.config = this->config;
            std::cout << " CONFIG: " << ret.config << std::endl;
            return ret;
        }

        // Create final TuningSession
        auto createSession()
        {
            TuningSession<T_Strategy, T_GridSize, T_BlockSize, grid, block> session{strategy};
            this->template copy<grid, block>(session); // Same copy logic
            session.reRuns = reRuns;
            return session;
        }
    };
} // namespace alpaka::tune
#endif // SESSIONBUILDER_H
