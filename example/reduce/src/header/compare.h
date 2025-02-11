//
// Created by tim on 14.01.25.
//

#ifndef COMPARE_H
#define COMPARE_H
//
// Created by tim on 04.01.25.
//
#ifndef REDUCEKERNEL_H
#define REDUCEKERNEL_H
#include "commonIncludes.hpp"
//! A reduce kernel.
// In standard projects, you typically do not execute the code with any available accelerator.
// Instead, a single accelerator is selected once from the active accelerators and the kernels are executed with the
// selected accelerator only. If you use the example as the starting point for your project, you can rename the
// example() function to main() and move the accelerator tag to the function body.
class Reduce
{
public:
    //! The kernel entry point.
    //!
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \param acc The accelerator to be executed on.
    //! \param type
    //! \param dataBuf The greyScale Image
    //! \param destinationBuf
    //! \param dataDomainExtent The number of elements.
    //! \param sharedMemExtents
    template<typename TAcc,typename operationType>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, const operationType &type,auto const dataBuf,auto destinationBuf, alpaka::concepts::Vector auto const dataDomainExtent,alpaka::concepts::CVector auto sharedMemExtents)-> void
    {
        using namespace alpaka;
        using namespace reduce;
        auto const blocksize=acc[alpaka::layer::thread].count();

        auto const gridSize=acc[alpaka::layer::block].count();
        auto const blockIdx=acc[alpaka::layer::block].idx();
        auto const threadIdx=acc[alpaka::layer::thread].idx().product();
        auto sdata = onAcc::declareSharedMdArray<float,uniqueId()>(acc, sharedMemExtents);

        auto numFrames = Vec<std::size_t,1>{acc[frame::count]};
        auto frameExtent = Vec<std::size_t,1>{acc[frame::extent]};

        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});
        /* init shared memory
         * no need to synchronize because after the loop because the same index map will be used to access the shared
         * memory in the tribble loop.
         */
        auto const frameDataExtent = numFrames * frameExtent;
        auto traverseOverFrames = onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{alpaka::Vec<std::size_t,1>{0}, frameDataExtent, frameExtent});
        for(auto fill : traverseInFrame)
        {
            sdata[fill]=static_cast<float>(0);
        }
        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdxInFrame : traverseInFrame)
            {

                for(auto [i] : onAcc::makeIdxMap(
                                        acc,
                                        onAcc::WorkerGroup{(frameIdx + elemIdxInFrame)*4, frameDataExtent},
                                        IdxRange{dataDomainExtent}))
                {
                    sdata[elemIdxInFrame]+=dataBuf[i];
                }
                    //reduce::operate<operationType,float>(sdata[threadIdx],dataBuf[elemIdxInFrame]);

                //printf(" bix: %li, access: %f, at index: %li \n",blockIdx,dataBuf[elemIdxInFrame],elemIdxInFrame);


            }

        }
        for(auto [elemIdxInFrame] :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(), frameExtent}))
        {
            sdata[acc[layer::thread].idx()] += sdata[elemIdxInFrame];
        }
        //std::cout<<" tiDx: "<<sdata[threadIdx]<<std::endl;
        constexpr auto one=alpaka::Vec{std::size_t{1}};
        constexpr auto two=alpaka::Vec{std::size_t{2}};
        //auto const stride = 16 / sizeof(DataType);
        //outer grid stride loop
        auto iters=0;
        constexpr alpaka::Vec<std::size_t,1> sMem=alpaka::Vec{static_cast<std::size_t>(sharedMemExtents.product())};
        //load from gmem to smem using 4 strided access (optimally enabling 128bit loads) -> 4 gmem entries get reduced to one smem entry
        auto step=alpaka::core::divCeil(blocksize,two).product();
        auto strideStep=1;
        onAcc::syncBlockThreads(acc);
        //printf("thread data: %f\n",sdata[acc[layer::thread].idx()]);
        for(auto dIdx :onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::threadsInBlock))
        {
            while(step>0)
            {
                if(threadIdx%step==0)
                {
                    sdata[threadIdx]=sdata[threadIdx]+sdata[threadIdx+strideStep];
                }
                strideStep<<=1;
                step>>=1;
                onAcc::syncBlockThreads(acc);
            }
        }
        if(threadIdx==0)
        {
            //std::cout<<"end : "<<sdata[0]<<std::endl;
            onAcc::atomicAdd(acc,static_cast<float*>(&destinationBuf[0]), sdata[0]);

            //onAcc::atomicAdd(&destinationBuf[0],static_cast<float>(sdata[0]));
        }





        //printf("\n");
    }
};
#endif //REDUCEKERNEL_H

#endif //COMPARE_H
