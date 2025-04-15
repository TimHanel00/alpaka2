
#ifndef SESSIONBUILDER_HPP
#define SESSIONBUILDER_HPP
#include "alpaka/onHost.hpp"
#include "alpaka/tune/active/KernelSingleton.hpp"
#include "alpaka/tune/utils/environmentVars.hpp"

#include <alpaka/tune/IO/tuningHistory.hpp>
#include <alpaka/tune/active/activeKernel.hpp>
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
    template<typename T_Strategy, typename... T_KernelRunArgs>
    struct TuningSession;
} // namespace alpaka

namespace alpaka::tune
{
    // forward declaration of TuningSession


    template<typename T_Strategy = strategy::randomSearch<Timing>, typename... T_KernelRunArgs>
    class TuningBuilder
    {
    public:
        TuningBuilder() = default;
        explicit TuningBuilder(ActiveKernelRun<T_KernelRunArgs...> const& newRun) : m_run(newRun) {};

        template<typename T_Tune>
        auto withTuning(T_Tune tuningObject) const
        {
            auto newRun = appendTuning(m_run, tuningObject);

            auto ret = helperCreateNewBuilder<T_Strategy>(newRun);
            ret.m_config = m_config;
            ret.m_reRuns = m_reRuns;
            ret.m_dynamicRuns = m_dynamicRuns;
            ret.m_sessionSpecifiers = m_sessionSpecifiers;
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
            std::cout << " in numblockstune" << std::endl;
            NumBlocksTune tuningObject{};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withNumBlocksTune(NumBlocksTune<T> tune) const
        {
            tune.userDef = true;
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withNumBlocksTune(alpaka::Vec<T, dim> tune) const
        {
            NumBlocksTune<decltype(tune)> tuningObject{tune};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        auto withBlockSizeTune() const
        {
            ThreadBlockSizeTune tuningObject{};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withBlockSizeTune(ThreadBlockSizeTune<T> tune) const
        {
            std::cout << " in blockSIzeTune" << std::endl;
            tune.userDef = true;
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withBlockSizeTune(alpaka::Vec<T, dim> tune) const
        {
            ThreadBlockSizeTune<decltype(tune)> tuningObject{tune};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        auto withNumFramesTune() const
        {
            NumFramesTune tuningObject{};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withNumFramesTune(NumFramesTune<T> tune) const
        {
            tune.userDef = true;
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withNumFramesTune(alpaka::Vec<T, dim> tune) const
        {
            NumFramesTune<decltype(tune)> tuningObject{tune};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        auto withFrameExtentTune() const
        {
            FrameExtentTune tuningObject{};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        template<typename T>
        auto withFrameExtentTune(FrameExtentTune<T> tune) const
        {
            tune.userDef = true;
            return this->withTuning(tune);
        }

        template<typename T, auto dim>
        auto withFrameExtentTune(alpaka::Vec<T, dim> tune) const
        {
            FrameExtentTune<decltype(tune)> tuningObject{tune};
            tuningObject.userDef = false;
            return this->withTuning(tuningObject);
        }

        // Output a fully constructed TuningSession
        auto build() const
        {
            return TuningSession<T_Strategy, T_KernelRunArgs...>(
                T_Strategy{},
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

    template<typename T_Strategy, typename... T_KernelRunArgs>
    auto helperCreateNewBuilder(ActiveKernelRun<T_KernelRunArgs...> const& newRun)
    {
        return TuningBuilder<T_Strategy, T_KernelRunArgs...>(newRun);
    }
}; // namespace alpaka::tune
#endif
