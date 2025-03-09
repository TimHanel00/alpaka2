//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include <vector>
#include <random>
namespace alpaka::tune::strategy
{
    template<typename TGridSize,typename TthreadBlockSize,typename tuneables,typename KernelRun>
    bool inHistory(TGridSize &gridSize,TthreadBlockSize & threadBlockSize,std::vector<tuneables> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history)
    {
        for (const auto& run : history)
        {
            if(gridSize==run->gridSize&&threadBlockSize==run->threadBlockSize)
            {

                bool acc=true;
                for(int index = 0;index<tuningParameters.size();index++)
                {
                    acc=acc&&(tuningParameters[index]==run->tuneables[index]);
                }
                if(acc)return true;
            }
        }
        return false;
    }
    template<typename T_Begin, typename T_End, typename T_Stride>
    T_Begin randomIdx(const IdxRange<T_Begin, T_End, T_Stride>& range)
    {
        // Alias the vector type for the result.
        using VecType = T_Begin;
        // Create a result vector.
        VecType result;
        // Assume that T_Begin has a static member T_dim (or use T_End::T_dim).
        constexpr std::size_t dim = VecType::T_dim;

        // Set up a random number generator.
        // (Using static so that the generator is not re-seeded on every call.)
        static std::random_device rd;
        static std::mt19937 gen(rd());

        for (std::size_t i = 0; i < dim; ++i)
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
    struct randomSearch
    {
        template<typename tuneables,typename KernelRun,typename T_Device,typename T_Exec>
        auto operator()(std::vector<std::shared_ptr<tuneables>> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history,T_Device const& device, T_Exec const& exec) const
        {
            for(auto & parameter : tuningParameters)
            {
                parameter->value=randomIdx(parameter->idxRange).x();

            }
        }
    };
    struct bestRecorded{
        template<typename tuneables,typename KernelRun,typename T_Device,typename T_Exec>
        auto operator()(std::vector<std::shared_ptr<tuneables>> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history,T_Device const& device, T_Exec const& exec) const {
            /*if(history.empty()){return;}
            auto selectedRun = history[0];

            for (const auto& run : history) {
                if (run->metric > selectedRun->metric) {
                    selectedRun = run;  // Update selectedRun to the run with the smaller metric
                }
            }
            std::copy(selectedRun->tuneables.begin(),selectedRun->tuneables.end(),tuningParameters.begin());*/
        }
    };
    struct initialValues{
        template<typename tuneables,typename KernelRun,typename T_Device,typename T_Exec>
         auto operator()(std::vector<std::shared_ptr<tuneables>> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history,T_Device const& device, T_Exec const& exec) const {
            return tuningParameters;
        }
    };
}
#endif //STRATEGY_HPP
