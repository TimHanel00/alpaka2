//
// Created by tim on 15.04.25.
//

#ifndef METRICINTERFACE_H
#define METRICINTERFACE_H
#include "alpaka/tune/IO/storageTypes.hpp"

#include <thread>

namespace alpaka::tune
{
    namespace detail
    {
        // Shared helper to perform Kruskal-Wallis comparison
        enum class Comparison
        {
            Less,
            Greater,
            Inconclusive
        };
        enum class returnComparison
        {
            HigherIsBetter,
            LowerIsBetter,
        };
    } // namespace detail

    namespace global
    {
        inline auto timingAccessor(std::optional<double_t> value = std::nullopt)
        {
            static double_t m_value = 0.0;
            if(value.has_value())
                m_value = value.value();
            return m_value;
        };
    } // namespace global

    namespace metricInterface
    {
        struct Timing
        {
            static constexpr detail::returnComparison returnComparison{detail::returnComparison::LowerIsBetter};
            std::chrono::high_resolution_clock::time_point startTime;

            template<typename T_KernelRun, typename T_FrameSpec>
            void start(T_KernelRun& kernelRun, T_FrameSpec& frame_spec)
            {
                startTime = std::chrono::high_resolution_clock::now();
            }

            template<typename T_KernelRun, typename T_FrameSpec>
            void end(T_KernelRun& kernelRun, T_FrameSpec& frame_spec)
            {
                auto endTime = std::chrono::high_resolution_clock::now();
                auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
                kernelRun.metric = static_cast<decltype(kernelRun.metric)>(timeDuration.count());
                global::timingAccessor(kernelRun.metric);
            }
        };

        struct Energy
        {
        };

        /*
         * example for how a occupance based definition could look like, (contains a bit of pseudo code on the api
         * calls)
         */
        template<typename RegisterAPI_CALL>
        struct Occupancy
        {
            static constexpr detail::returnComparison returnComparison{detail::returnComparison::HigherIsBetter};
            RegisterAPI_CALL& apiCaller;
            std::thread workerThread;
            std::atomic<bool> stopFlag{false};
            std::vector<double_t> occupancy{};
            Occupancy(RegisterAPI_CALL& handle) : apiCaller(handle) {};

            template<typename T_KernelRun, typename T_FrameSpec>
            void start(T_KernelRun& kernelRun, T_FrameSpec& frame_spec)
            {
                stopFlag = false;
                workerThread = std::thread(
                    [this]()
                    {
                        while(!stopFlag.load())
                        {
                            occupancy.emplace_back(apiCaller.call());
                            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // tune as needed
                        }
                    });
            }

            template<typename T_KernelRun, typename T_FrameSpec>
            void end(T_KernelRun& kernelRun, T_FrameSpec& frame_spec)
            {
                stopFlag = true;
                if(workerThread.joinable())
                {
                    workerThread.join();
                }
            }
        };
    } // namespace metricInterface

    namespace concepts
    {
        template<typename T>
        concept MetricInterface = requires(T t) {
            { t.returnComparison } -> std::convertible_to<detail::returnComparison>;
            {
                t.start(
                    std::declval<KernelTuningModel<>&>(),
                    std::declval<onHost::FrameSpec<Vec<uint32_t, 1>, Vec<uint32_t, 1>>&>())
            } -> std::same_as<void>;
            {
                t.end(
                    std::declval<KernelTuningModel<>&>(),
                    std::declval<onHost::FrameSpec<Vec<uint32_t, 1>, Vec<uint32_t, 1>>&>())
            } -> std::same_as<void>;
            //{ t.end(std::declval<R&>(), std::declval<S&>()) } -> std::same_as<void>;
        }; // namespace concepts
    } // namespace concepts

    // wraps any type of metric (usually double) and adds overloads according to
    template<typename T_Metric>
    static auto compareGetBest(T_Metric metric, auto&& a, auto&& b)
        requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
    {
        if(a > b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto compareGetBest(T_Metric metric, auto&& a, auto&& b)
        requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
    {
        if(a < b)
            return a;
        return b;
    }

    template<typename T_Metric, typename T_ConfigEntry>
    struct aGTb
    {
        T_ConfigEntry& operator()(T_ConfigEntry& a, T_ConfigEntry& b)
        {
            if constexpr(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
            {
                return b;
            }
            else
            {
                return a;
            }
        }
    };

    template<typename T_Metric, typename T_ConfigEntry>
    struct aLTb
    {
        T_ConfigEntry& operator()(T_ConfigEntry& a, T_ConfigEntry& b)
        {
            if constexpr(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
            {
                return b;
            }
            else
            {
                return a;
            }
        }
    };

    namespace strategy::SimulatedAnnealing
    {
        template<typename T_Metric, typename T_ConfigEntry>
        struct costDifference
        {
            auto operator()(T_ConfigEntry& a, T_ConfigEntry& b)
                requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
            {
                // Default: assume higher is better
                return a.template getMetric<median_t>().template as<t_ns>()
                       - b.template getMetric<median_t>().template as<t_ns>();
            }

            auto operator()(T_ConfigEntry& a, T_ConfigEntry& b)
                requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
            {
                // Default: assume lower is better
                return b.template getMetric<median_t>().template as<t_ns>()
                       - a.template getMetric<median_t>().template as<t_ns>();
            }
        };
    } // namespace strategy::SimulatedAnnealing


} // namespace alpaka::tune
#endif // METRICINTERFACE_H
