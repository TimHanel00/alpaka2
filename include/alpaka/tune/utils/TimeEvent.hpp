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

    template<typename T_KernelRun>
    static TimeEvent<T_KernelRun> createTimeEvent(T_KernelRun& run)
    {
        return TimeEvent(run);
    }

    template<typename T_ActiveKernelRun>
    static auto createTimeEventFromActive(T_ActiveKernelRun& activeRun)
    {
        return TimeEvent(activeRun);
    }
} // namespace alpaka::tune
#endif // TIMEEVENT_H
