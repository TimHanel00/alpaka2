/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engines/philox/PhiloxStateless.hpp"

#include <alpaka/rand/engines/philox/PhiloxState.hpp>

namespace alpaka::rand::internal::engine
{
    /** Common class for Philox family engines
     *
     * Relies on `PhiloxStateless` to provide the PRNG and adds state to handling the counting.
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     * @tparam TImpl engine type implementation (CRTP)
     *
     * static const data members are transformed into functions, because GCC
     * assumes types with static data members to be not mappable and makes not
     * exception for constexpr ones. This is a valid interpretation of the
     * OpenMP <= 4.5 standard. In OpenMP >= 5.0 types with any kind of static
     * data member are mappable.
     */
    template<typename TParams, typename TImpl>
    class PhiloxBaseCommon : public PhiloxStateless<TParams>
    {
    public:
        using Counter = typename PhiloxStateless<TParams>::Counter;
        using Key = typename PhiloxStateless<TParams>::Key;
        /// State type
        using State = PhiloxState<Counter, Key>;

        /// Internal engine state
        State state;
        /// Distribution container type
        template<typename TDistributionResultScalar>
        using ResultContainer = typename alpaka::Vec<TDistributionResultScalar, TParams::counterSize>;
        ALPAKA_FN_HOST_ACC
        PhiloxBaseCommon() = default;

        ALPAKA_FN_HOST_ACC
        PhiloxBaseCommon(Counter c, Key k, Counter r, std::uint32_t n = 0u)
            : state{std::move(c), std::move(k), std::move(r), n}
        {
        }

    protected:
        /** Advance the \a counter to the next state
         *
         * Increments the passed-in \a counter by one with a 128-bit carry.
         *
         * @param counter reference to the counter which is to be advanced
         */
        template<typename T, auto N>
        ALPAKA_FN_HOST_ACC void advanceCounter(alpaka::Vec<T, N>& counter)
        {
            for(auto i = 0; i < N; ++i)
            {
                if(++counter[i] != 0)
                    break;
            }
        }

        /** Advance the internal state counter by \a offset N-vectors (N = counter size)
         *
         * Advances the internal value of this->state.counter
         *
         * @param offset number of N-vectors to skip
         */
        ALPAKA_FN_HOST_ACC void skip4(uint64_t offset)
        {
            Counter& counter = this->state.counter;
            Counter temp = counter;
            counter[0] += helper::low32Bits(offset);
            counter[1] += helper::high32Bits(offset) + (counter[0] < temp[0] ? 1 : 0);
            counter[2] += (counter[0] < temp[1] ? 1u : 0u);
            counter[3] += (counter[0] < temp[2] ? 1u : 0u);
        }

        /** Advance the counter by the length of \a subsequence
         *
         * Advances the internal value of this->state.counter
         *
         * @param subsequence number of subsequences to skip
         */
        ALPAKA_FN_HOST_ACC void skipSubsequence(uint64_t subsequence)
        {
            Counter& counter = this->state.counter;
            Counter temp = counter;
            counter[2] += helper::low32Bits(subsequence);
            counter[3] += helper::high32Bits(subsequence) + (counter[2] < temp[2] ? 1 : 0);
        }
    };
} // namespace alpaka::rand::internal::engine
