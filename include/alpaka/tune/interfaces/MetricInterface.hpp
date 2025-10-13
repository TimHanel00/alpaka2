//
// Created by tim on 15.04.25.
//

#ifndef METRICINTERFACE_H
#define METRICINTERFACE_H
// #include "alpaka/tune/IO/storageTypes.hpp"

#include <thread>

namespace alpaka::tune
{
    namespace detail
    {
        // Shared helper to perform Kruskal-Wallis comparison
        enum class returnComparison
        {
            HigherIsBetter,
            LowerIsBetter,
        };
    } // namespace detail

    namespace metricInterface
    {
        struct Timing
        {
            static constexpr detail::returnComparison returnComparison{detail::returnComparison::LowerIsBetter};
            std::chrono::high_resolution_clock::time_point startTime;

            void start()
            {
                startTime = std::chrono::high_resolution_clock::now();
            }

            auto end() -> double_t
            {
                auto endTime = std::chrono::high_resolution_clock::now();
                auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
                return timeDuration.count();
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

            auto end() -> double_t
            {
                stopFlag = true;
                if(workerThread.joinable())
                {
                    workerThread.join();
                }
                return apiCaller.handle().getAchievedOccupancy;
            }
        };
    } // namespace metricInterface

    namespace concepts
    {
        template<typename T>
        concept MetricInterface = requires(T t) {
            { t.returnComparison } -> std::convertible_to<detail::returnComparison>;
            { t.start() } -> std::same_as<void>;
            { t.end() } -> std::same_as<double_t>;
            //{ t.end(std::declval<R&>(), std::declval<S&>()) } -> std::same_as<void>;
        }; // namespace concepts
    } // namespace concepts

    // wraps any type of metric (usually double) and adds overloads according to
    template<typename T_Metric>
    static auto const& compareGetBest(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
    {
        if(a > b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetWorst(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
    {
        if(a < b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetBest(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
    {
        if(a < b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetWorst(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
    {
        if(a > b)
            return a;
        return b;
    }

    namespace strategy::SimulatedAnnealing
    {
        template<typename T_Metric, typename T_ConfigEntry>
        struct costDifference
        {
            auto operator()(T_ConfigEntry const& a, T_ConfigEntry const& b)
                requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
            {
                // assume higher is better
                return a.getMedian() - b.getMedian();
            }

            auto operator()(T_ConfigEntry const& a, T_ConfigEntry const& b)
                requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
            {
                // assume lower is better flip the sign -- writing it like this make the intend more obvious
                return -(a.getMedian() - b.getMedian());
            }
        };
    } // namespace strategy::SimulatedAnnealing


} // namespace alpaka::tune
#endif // METRICINTERFACE_H
