//
// Created by tim on 05.02.25.
//
#define ENABLE_AUTOTUNE
#ifdef ENABLE_AUTOTUNE
#ifndef TUNER_H
#define TUNER_H
#include <yaml-cpp/yaml.h>
#include <alpaka/onHost/mem/Handle.hpp>

namespace alpaka::onHost::tune{
    #define TUNEALL 230
    #define TUNEGRID 1237
    #define TUNETHREADBLOCK 1328
    template<typename
    struct KernelData{
         ;
    }
    class Tuner{
        public:
            yaml_data=None;
            Tuner(){
                yaml_data=loadYamlFile("tune.yaml");
                }
            yamlData=NULL;
            ~Tuner(){
                write(yamlData to file);
            };
            bool tuneGrid=0;
            bool tuneBlock=0;
            bool load_config=0;
            unordered_map<
            auto getBlockSize(auto dataDomain, auto Accelorator){
                if(!load_config){
                    loadConfig();
                }
                if(yamlData==NULL){
                    if(yamlData.count(Accelerator.id)!=0&&yamlData[Accelerator.id].count(dataDomain.getExtents().product())!=0)!=0;
                    auto cfg = yamlData[Accelerator.id][dataDomain.getExtents().product()];
                    return cfg[0].as<std::uint32_t>();
                };
            }
            auto getBlockSize(auto dataDomain, auto Accelorator){
                if(!load_config){
                    loadConfig();
                }
                if(yamlData==NULL){
                    if(yamlData.count(Accelerator.id)!=0&&yamlData[Accelerator.id].count(dataDomain.getExtents().product())!=0)!=0;
                    auto cfg = yamlData[Accelerator.id][dataDomain.getExtents().product()];
                    return cfg[0].as<std::uint32_t>();
                };
            }
            auto insertYamlData(auto kernelCFG){
                }
            auto loadConfig(){
                if(constexpr std::string filename exist){
                    yamlData=yaml load filename
                }

       template<typename Platform,typename T_Mapping,
    typename T_NumBlocks,
    typename T_NumThreads,
    typename T_KernelBundle> auto tune(unifiedCudaHip::Device<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>{
        //gpu specific code
        };
    template<typename Platform,typename T_Mapping,
    typename T_NumBlocks,
    typename T_NumThreads,
    typename T_KernelBundle> auto tune(cpu::Device<<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>{
        //cpu specific code
        };
    }
    template<typename T_Device,
        typename T_Mapping,
        typename T_NumBlocks,
        typename T_NumThreads,
        typename T_KernelBundle>
        auto tune(
                T_Device const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle){
                static Tuner tuner();
                //Tuning steps that are general
                return threadSpec=tuner.tune(device,executor,dataBlocking,kernelBundle);

            }
}
#endif //TUNER_H
#endif