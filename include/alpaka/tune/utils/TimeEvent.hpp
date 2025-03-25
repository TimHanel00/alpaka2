//
// Created by tim on 19.03.25.
//

#ifndef TIMEEVENT_H
#define TIMEEVENT_H
#include <chrono>

namespace alpaka::tune
{
    template<typename T_KernelRun>
    struct TimeEvent
    {
        std::chrono::high_resolution_clock::time_point startTime;
        T_KernelRun& kernelRun;

        explicit TimeEvent(T_KernelRun& run) : startTime(std::chrono::high_resolution_clock::now()), kernelRun(run)
        {
        }

        // Destructor: Stops the timer and records the duration
        ~TimeEvent()
        {
            auto const endTime = std::chrono::high_resolution_clock::now();
            auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
            kernelRun.metric = convertToT<ALPAKA_TYPEOF(kernelRun.metric)>(timeDuration.count());
        }
    };

    template<typename T_TuningSession>
    static auto createTimeEvent(T_TuningSession& session)
    {
        return TimeEvent(session.kernelRun);
    }

    template<typename T_grid, typename T_block, typename T_Tuneables>
    static auto createTimeEventFromActive(ActiveKernelRun<T_grid, T_block, T_Tuneables>& activeRun)
    {
        return TimeEvent(activeRun);
    }
} // namespace alpaka::tune
#endif // TIMEEVENT_H
