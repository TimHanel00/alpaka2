//
// Created by tim on 15.04.25.
//

#ifndef METRICINTERFACE_H
#define METRICINTERFACE_H
#include "alpaka/tune/IO/storageTypes.hpp"

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
    } // namespace detail

    struct Timing
    {
    };

    struct Energy
    {
    };

    struct Occupancy
    {
    };

    struct MetricAdjust
    {
        template<typename T_Metric>
        struct aGTb
        {
            StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b) const
            {
                // Default: assume higher is better
                return a;
            }
        };

        template<typename T_Metric>
        struct aLTb
        {
            StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b) const
            {
                // Default: assume higher is better
                return b;
            }
        };

        template<typename T_Metric>
        struct costDifference
        {
            auto operator()(StorageKernelRun& a, StorageKernelRun& b) const
            {
                // Default: assume higher is better
                return a.getMetric<median_t>().as<t_ns>() - b.getMetric<median_t>().as<t_ns>();
            }
        };
    };

    /*
     * since time is currently no defined interface this is used to compare time metrics
     * */
    template<>
    struct MetricAdjust::aGTb<Timing>
    {
        StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b)
        {
            return b;
        }
    };

    template<>
    struct MetricAdjust::aLTb<Timing>
    {
        StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b)
        {
            return a;
        }
    };

    // Specialize costDifference<double_t>
    template<>
    struct MetricAdjust::costDifference<Timing>
    {
        auto operator()(StorageKernelRun& old_, StorageKernelRun& new_)
        {
            return old_.getMetric<median_t>().as<t_ns>() - new_.getMetric<median_t>().as<t_ns>();
        }
    };

    template<>
    struct MetricAdjust::aGTb<Occupancy>
    {
        StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b)
        {
            return a;
        }
    };

    template<>
    struct MetricAdjust::aLTb<Occupancy>
    {
        StorageKernelRun& operator()(StorageKernelRun& a, StorageKernelRun& b)
        {
            return b;
        }
    };

    // Specialize costDifference<double_t>
    template<>
    struct MetricAdjust::costDifference<Occupancy>
    {
        auto operator()(StorageKernelRun& old_, StorageKernelRun& new_)
        {
            return new_.getMetric<median_t>().as<t_ns>() - old_.getMetric<median_t>().as<t_ns>();
        }
    };


} // namespace alpaka::tune
#endif // METRICINTERFACE_H
