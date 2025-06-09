/* Copyright 2025 Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#if ALPAKA_LANG_ONEAPI

#    include "alpaka/Vec.hpp"
#    include "alpaka/core/Assert.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/core/Vectorize.hpp"
#    include "alpaka/tag.hpp"

#    include <sycl/sycl.hpp>

#    include <functional>

namespace alpaka::onAcc
{
    namespace oneApi
    {
        namespace detail
        {
            struct MetaData
            {
                //! pointer to allocated data
                uint8_t* ptr = nullptr;
                //! Unique id if the next data chunk.
                size_t id = std::numeric_limits<size_t>::max();
            };
            static constexpr uint32_t metaDataSize = sizeof(MetaData);
            /** number of bytes required for bookkeeping of maxNumberOfAllocations unique allocations
                 *
                 * @param maxNumUniqueAllocations number of unique allocation a user is allowed to perform
                 * @return bytes required to store lookup meta data
                 */
            static consteval uint32_t sizeLookupBufferInBytes(uint32_t maxNumUniqueAllocations)
            {
                return metaDataSize * maxNumUniqueAllocations;
            }

            /** Pointer lookup table
             *
             * Provides a dynamic lookup table to map an unique id to a pointer.
             */
            template<auto Dim>
            class PtrLookupTable
            {




            public:
#    ifndef NDEBUG
                PtrLookupTable(std::uint8_t* mem, uint32_t capacity, sycl::nd_item<Dim> const& item)
                : m_mem(reinterpret_cast<MetaData*>(mem))
                , m_capacity(capacity / metaDataSize)
                , item(item)
                {
                    ALPAKA_ASSERT_ACC((m_mem == nullptr) == (m_capacity == 0u));
                }
#    else
                PtrLookupTable(std::uint8_t* mem, uint32_t,sycl::nd_item<1> const &_item) : m_mem(reinterpret_cast<MetaData*>(mem)),item(_item)
                {
                }
#    endif


                template<typename T>
                T* alloc(size_t id) const
                {
                    auto group = item.get_group();
                    T* data = sycl::ext::oneapi::group_local_memory_for_overwrite<T>(group);

                    MetaData& metaDataEntry = m_mem[m_numEntries];
                    ++m_numEntries;
                    ALPAKA_ASSERT_ACC(m_numEntries <= m_capacity);

                    // Update meta data with id and pointer to the current allocation
                    if(group.get_local_linear_id() == 0u)
                    {
                        // only one thread must update the pointer in shared memory
                        metaDataEntry.ptr = reinterpret_cast<uint8_t*>(data);
                    }
                    metaDataEntry.id = id;

                    return data;
                }

                //! Give the pointer to an exiting variable
                //!
                //! @tparam T type of the variable
                //! @param id unique id of the variable
                //! @return nullptr if variable with id not exists
                template<typename T>
                auto getVarPtr(size_t id) const -> T*
                {
                    // Iterate over metadata
                    for(uint32_t off = 0u; off < m_numEntries; ++off)
                    {
                        MetaData& metaDataEntry = m_mem[off];

                        if(metaDataEntry.id == id)
                            return reinterpret_cast<T*>(metaDataEntry.ptr);
                    }

                    // Variable not found.
                    return nullptr;
                }

            private:
                sycl::nd_item<Dim> const & item;
                //! Number unqiue meta data entries stored
                mutable uint32_t m_numEntries = 0u;

                //! Memory layout
                //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
                //! Size of padding can be zero if data after padding is already aligned.
                MetaData* const m_mem;
#    ifndef NDEBUG
                //! max number of meta data entries
                uint32_t const m_capacity;
#    endif
            };
        } // namespace detail
        template<auto Dim>
        class StaticSharedMemory : private detail::PtrLookupTable<Dim>
        {
        public:
            StaticSharedMemory(StaticSharedMemory const&) = delete;
            using Base = detail::PtrLookupTable<Dim>;
            /** Construct shared memory allocator
             * @param accessor local memory accessor to store lookup meta data
             *                 bytes required to store N unique allocation can be calculated with
             * sizeLookupBufferInBytes()
             */
            StaticSharedMemory(sycl::local_accessor<std::byte> const& accessor,sycl::nd_item<Dim> const& item)
                : Base(
                    reinterpret_cast<std::uint8_t*>(accessor.get_multi_ptr<sycl::access::decorated::no>().get()),
                    static_cast<uint32_t>(accessor.size()),item)

            {
            }



            template<typename T, size_t T_unique>
            T& allocVar()
            {
                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    data = Base::template alloc<T>(T_unique);
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }
        };

    } // namespace oneApi
} // namespace alpaka::onAcc

#endif
