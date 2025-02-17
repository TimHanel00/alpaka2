/* Copyright 2024 Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber, Jan Stephan, Luca Ferragina,
 *                Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: ISC
 */
#include "header/startKernelWrapper.hpp"
#include "header/functor.hpp"
#include "header/reduceKernel.hpp"
#include "header/utils.hpp"

#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <typeinfo>
using IdxType=std::size_t;
using namespace alpaka;

template<uint32_t T_dim>
using VecDev = alpaka::Vec<uint32_t, T_dim>;
float error=0;
double initFlops=0;
std::size_t bufSize=0;
namespace examples::Reduce
{
    /*initializes input array and returns expected result
     *
     */
    template<typename operationType,typename Data,typename bufHost>
    std::vector<Data> initReduce(bufHost& A){


        for(IdxType i(0);i<A.getExtents().product();i++)
        {
            auto lIdx = alpaka::mapToND(A.getExtents(), i);
            A.getMdSpan()[lIdx] = static_cast<Data>(i+1);
        }
        auto firstIndex = alpaka::mapToND(A.getExtents(), static_cast<IdxType>(0));
        Data acc = A.getMdSpan()[firstIndex];
        auto const initbeginT = std::chrono::high_resolution_clock::now();
        for(IdxType i(1);i<A.getExtents().product();i++)
        {
            //auto lIdx = alpaka::mapToND(A.getExtents(), i);
            acc=reduce::operate(operationType{},acc,A.getMdSpan()[i]);
            //acc=Reduce::;
        }
        auto n=static_cast<Data>(A.getExtents().product());

        auto const endTEndT = std::chrono::high_resolution_clock::now();
        initFlops=(A.getExtents().product()/std::chrono::duration<double>(endTEndT - initbeginT).count())/1e9;
        bufSize=A.getExtents().product();
        return {acc,(n*(n+1))/2};//gauschen Formula for validation is returned
    }
    /*  actualLowerBound = for loop accumulated value
     *  actualUpperBound = gaussschen approximation ((n*(n+1))/2)
     */
    template<typename Data>
   int validate(Data transfer,Data actualLowerBound, Data actualUpperBound)
    {
        error=std::fabs(transfer-actualUpperBound);
        if(std::fabs(transfer-actualUpperBound)<1e-9||transfer==actualUpperBound||transfer==actualLowerBound)
        {
            return 0;
        }

        if(actualUpperBound < actualLowerBound)
        {
            std::swap(actualLowerBound,actualUpperBound);
        }

        if(transfer>actualLowerBound&&transfer<actualUpperBound)
        {
            return 0;
        }
        std::cout<<"error "<<bufSize<<std::endl;
        std::cout<<"device: "<<transfer<<"vs validate: " <<actualUpperBound<<std::endl;
        return 0;
    }

}

template<typename T_Cfg>
auto example(T_Cfg const& cfg) -> int
{
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];
    using dType=float_t;
    //makeAcc
    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);
    //specify nr of run12

    std::size_t nr_of_Runs=12;
    std::vector<std::size_t> data={64};
    //std::vector<std::size_t> data={65536};

    for(auto i=1;i<nr_of_Runs;i++)
    {
        data.push_back(data[i-1]*4);
    }
    for(std::size_t & i : data)
    {

        const auto extent=alpaka::Vec<std::size_t,1u>{i};
        {
            auto bufHostA = onHost::alloc<dType>(devHost, extent);
            std::vector<dType> valData=examples::Reduce::initReduce<reduce::sum,dType>(bufHostA);
            auto const beginT = std::chrono::high_resolution_clock::now();
            //reduction call
            auto constexpr numLoads=4;
            auto constexpr stride=4;
            auto result=reduction<4,4>(reduce::sum{} ,exec,devHost,devAcc,bufHostA);
            examples::Reduce::validate( result,valData[0],valData[1]);
        }


    }
    return 0;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEach(
        [=](auto const& tag) { return example(tag); },
        onHost::allExecutorsAndApis(onHost::enabledApis));
}