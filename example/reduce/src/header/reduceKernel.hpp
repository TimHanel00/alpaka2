//
// Created by tim on 04.01.25.
//
#ifndef REDUCEKERNEL_H
#define REDUCEKERNEL_H
#include "commonIncludes.hpp"
#include "functor.hpp"
template<typename Td,std::size_t strideSize>
struct alignas(sizeof(Td)*strideSize) mystructa
{
    float a[strideSize];
};

/*
 *implements a dynamic binary tree reduce
 */
template <typename operationType, typename T, std::size_t... Indices>
ALPAKA_FN_INLINE ALPAKA_FN_ACC T dynamicReduce(const operationType &type,const T* elements, std::index_sequence<Indices...>) {
    if constexpr (sizeof...(Indices) == 1) {
        return elements[0];
    } else {
        // Reduce pairwise in a binary form
        return reduce::operate(type,
            dynamicReduce(type,elements, std::make_index_sequence<sizeof...(Indices) / 2>{}),
            dynamicReduce(type,elements + sizeof...(Indices) / 2, std::make_index_sequence<sizeof...(Indices) / 2>{})
        );
    }
}

template<typename iter,typename IdxType>
ALPAKA_FN_INLINE ALPAKA_FN_ACC auto advance(iter &it,IdxType idx)
{
    return (*it++).x();
}
/*
 *
 */
template <std::size_t strideSize = 4, typename Td,typename operationType, typename Buftype, typename IndexType>
ALPAKA_FN_INLINE ALPAKA_FN_ACC  auto accFunctor(const operationType &type,Buftype& buf, IndexType index) -> Td
{
    using chunkType=mystructa<Td,strideSize>;

    const chunkType* alignedBuf = reinterpret_cast<const chunkType*>(buf.data()) + (index / strideSize);
    chunkType chunk = alignedBuf[0];
    Td* elements = reinterpret_cast<Td*>(&chunk);
    return dynamicReduce(type,elements, std::make_index_sequence<strideSize>{});
    //return reduce::operate<operationType>(reduce::operate<operationType>(elements[0], elements[1]),reduce::operate<operationType>(elements[2], elements[3]));//this is now hardcoded with a stride of 4

}
/*
 * performs multiple Load operations at the various data indices given by Index sequence
 * in the attempt to guarentee compile time loop unrolling and lower register count
 *
 */
template < std::size_t stride, typename float_t,typename operationType, typename Buftype, typename Iter, std::size_t... Indices>
ALPAKA_FN_INLINE ALPAKA_FN_ACC float_t multiLoadUnrolling(
    const operationType &type,Buftype& dataBuf, Iter& iter, std::index_sequence<Indices...>) {
    // Array to store intermediate values from accFunctor
    float_t results[sizeof...(Indices)] = {
        accFunctor<stride, float_t>(type,dataBuf, advance(iter,Indices))...
    };

    // Perform hierarchical reduction on the results array
    return dynamicReduce(type,
        results, std::make_index_sequence<sizeof...(Indices)>{});
}
#include <iomanip>
template<std::size_t stride,std::size_t numLoads,typename T>
class Reduce
{
public:
    uint32_t dynSharedMemBytes = 256u*4u;

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
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, const operationType type,auto const dataBuf,auto destinationBuf,auto const start,auto const end)-> void
    {
        using namespace alpaka;
        using namespace reduce;

        using IdxType=std::remove_reference_t<decltype(dataBuf.getExtents().x())>;
        using IdxVec = std::remove_reference_t<decltype(dataBuf.getExtents())>;
        using DataType = std::remove_const_t<std::remove_reference_t<decltype(dataBuf[0])>>;
        auto const dataDomainExtent=dataBuf.getExtents();
        auto numFrames = IdxVec{acc[frame::count]};

        auto frameExtent = IdxVec{acc[frame::extent]};
        //printf("threadCount: %li%, frameExtent: %li%\n",acc[layer::thread].count().x(), frameExtent.x());
        auto sdata = onAcc::getDynSharedMem<DataType>(acc);
        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});

        auto const blockDataExtent = acc[alpaka::layer::block].count() * frameExtent;
        auto const frameDomainExtent = numFrames * frameExtent;
        auto traverseOverFrames= onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{numFrames});
        auto traverseOverBlocks = onAcc::makeIdxMap(
            acc,
            onAcc::worker::blocksInGrid,
            IdxRange{IdxVec{0}, blockDataExtent, frameExtent});
        for(auto elemIdxInFrame : traverseInFrame)
        {
            sdata[elemIdxInFrame.x()]=reduce::neutral_element<DataType>(type);
        }
        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdxInFrame : traverseInFrame)
            {
                auto innerWorkgroup=onAcc::WorkerGroup{frameIdx*frameExtent + elemIdxInFrame, frameDomainExtent};
                onAcc::forEach<64>(
                    acc,
                    innerWorkgroup,
                    IdxVec{dataDomainExtent},
                    [&](auto const&, auto&& l_a) constexpr
                    {
                        //printf("%li sum \n",sum);
                        sdata[elemIdxInFrame.x()] +=l_a.load().sum();
                    },
                    dataBuf);

            }

        }

        onAcc::syncBlockThreads(acc);

        for(auto [DataElemIdxInFrame] :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(),frameExtent.x()}))
            {
                sdata[acc[layer::thread].idx().x()]=reduce::operate(type,sdata[acc[layer::thread].idx().x()],sdata[DataElemIdxInFrame]);
            }

        onAcc::syncBlockThreads(acc);
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            onAcc::syncBlockThreads(acc);
            if(local_i < offset)
            {
                sdata[local_i]=operate(type,sdata[local_i],sdata[local_i + offset]);
            }
        }



        onAcc::syncBlockThreads(acc);
        if(acc[layer::thread].idx().product()==0)
        {

            reduce::atomicOp(acc,type,destinationBuf.data(), sdata[acc[layer::thread].idx().x()]);
        }

    }
};
#endif //REDUCEKERNEL_H
