//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include <vector>

namespace alpaka::tune::strategy
{
    struct bestRecorded{
        template<typename TGridSize,typename TthreadBlockSize,typename tuneables,typename KernelRun,typename T_Device,typename T_Exec>
        auto operator()(TGridSize &gridSize=std::nullopt,TthreadBlockSize & threadBlockSize=std::nullopt,std::vector<tuneables> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history,T_Device const& device, T_Exec const& exec) const {
            if(history.empty()){return;}
            auto selectedRun = history[0];

            for (const auto& run : history) {
                if (run->metric > selectedRun->metric) {
                    selectedRun = run;  // Update selectedRun to the run with the smaller metric
                }
            }
            tuningParameters= std::copy(selectedRun->tuneables.begin(),selectedRun->tuneables.end());
        }
    };
    struct initialValues{
        template<typename TGridSize,typename TthreadBlockSize,typename tuneables,typename KernelRun,typename T_Device,typename T_Exec>
         auto operator()(TGridSize &gridSize=std::nullopt,TthreadBlockSize & threadBlockSize=std::nullopt,std::vector<tuneables> &tuningParameters,std::vector<std::shared_ptr<KernelRun>> history,T_Device const& device, T_Exec const& exec) const {
            return tuningParameters;
        }
    };
}
#endif //STRATEGY_HPP
