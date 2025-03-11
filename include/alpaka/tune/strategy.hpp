//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include <vector>
#include <random>
#include "tuner.hpp"
namespace alpaka::tune::strategy
{
    template<typename T_Begin, typename T_End, typename T_Stride>
    T_Begin randomIdx(const IdxRange<T_Begin, T_End, T_Stride>& range)
    {
        // Alias the vector type for the result.
        using VecType = T_Begin;

        // Create a result vector.
        VecType result;
        // Assume that T_Begin has a static member T_dim (or use T_End::T_dim).
        constexpr auto dim = IdxRange<T_Begin, T_End, T_Stride>::dim();

        // Set up a random number generator.
        // (Using static so that the generator is not re-seeded on every call.)
        static std::random_device rd;
        static std::mt19937 gen(rd());

        for (std::size_t i = 0; i < static_cast<std::size_t>(dim); ++i)
        {
            // For each dimension, retrieve the minimum, maximum and stride.
            auto minVal  = range.m_begin[i];
            auto maxVal  = range.m_end[i];
            auto step    = range.m_stride[i];

            // Calculate how many discrete steps we have.
            // This assumes that (maxVal - minVal) is an exact multiple of step.
            auto numSteps = (maxVal - minVal) / step;

            // If there are no steps (or only one valid value), use minVal.
            if(numSteps <= 0)
            {
                result[i] = minVal;
            }
            else
            {
                // Choose a random step index between 0 and numSteps - 1.
                std::uniform_int_distribution<decltype(minVal)> dis(0, numSteps - 1);
                auto k = dis(gen);
                // Set the i-th component as minVal + k * step.
                result[i] = minVal + k * step;
            }
        }
        return result;
    }
    template<typename T,typename startIdx,typename T_Begin, typename T_End, typename T_Stride>
    T getNextUpper(const T &value,const startIdx &idx,const IdxRange<T_Begin, T_End, T_Stride>& range,bool &valid)
    {
        using VecType = T_Begin;
        // Create a result vector.
        VecType result{value};
        auto i = static_cast<std::size_t>(idx);
        auto minVal  = range.m_begin[i];
        auto maxVal  = range.m_end[i];
        auto step    = range.m_stride[i];
        result[i]=(result[i]+step);
        if(value>maxVal)
        {
            valid=false;
            return value;
        }

        return T(result.product());
    }
    template<typename T,typename startIdx,typename T_Begin, typename T_End, typename T_Stride>
    T getNextLower(const T &value,const startIdx &idx,const IdxRange<T_Begin, T_End, T_Stride>& range, bool &valid)
    {
        using VecType = T_Begin;
        // Create a result vector.
        VecType result{value};
        auto i = static_cast<std::size_t>(idx);
        auto minVal  = range.m_begin[i];
        auto maxVal  = range.m_end[i];
        auto step    = range.m_stride[i];
        result[i]=(result[i]+step);
        if(value<minVal)
        {
            valid=false;
            return value;
        }

        return T(result.product());
    }
    struct randomSearch
    {
        template<typename tuneables,typename T_KernelRun,typename KernelRun>
        auto operator()(std::vector<std::shared_ptr<tuneables>> &tuningParameters,T_KernelRun &kernelRun,std::unordered_map<std::string,KernelRun> &history) const
        {
            int index=0;
            for(auto & parameter : tuningParameters)
            {
                parameter->value=randomIdx(parameter->idxRange).x();
                index++;
            }
            if(history.contains(kernelRun.toHash()))
            {
                for(auto & parameter : tuningParameters)
                {
                    constexpr auto dim=static_cast<std::size_t>(1);
                    //@TODO make this dynamic but ALPAKA_TYPE_OF(parameter->idxRange)::dim() did no get deduced correctly on GPU
                    using type=std::size_t;
                    auto initialValue=parameter->value;
                    for(auto i= static_cast<type>(0);i<dim; ++i)
                    {
                        bool valid=true;
                        while(valid)
                        {

                            parameter->value=getNextUpper(parameter->value,i,parameter->idxRange,valid);
                            if(!history.contains(kernelRun.toHash()))return;
                        }
                        parameter->value=initialValue;
                        valid=true;
                        while(valid)
                        {
                            parameter->value=getNextLower(parameter->value,i,parameter->idxRange,valid);
                            if(!history.contains(kernelRun.toHash()))return;
                        }

                    }


                }

            }
        };
    };
    //@TODO move to different namespace
    struct bestRecorded{
        template<typename T_KernelRun,typename KernelRun>
        auto operator()(T_KernelRun &kernelRun,std::unordered_map<std::string,KernelRun> &history) const
        {
            kernelRun = history.begin()->second;

            for (auto& run : history) {
                if (run.second.metric < kernelRun.metric) {
                    kernelRun = run.second;  // Update selectedRun to the run with the smaller metric
                }
            }
        }
    };
    struct initialValues{
        template<typename tuneables,typename T_KernelRun,typename KernelRun>
        auto operator()(std::vector<std::shared_ptr<tuneables>> &tuningParameters,T_KernelRun &kernelRun,std::unordered_map<std::string,KernelRun> &history) const
        {
            return tuningParameters;
        }
    };
}
#endif //STRATEGY_HPP
