//
// Created by tim on 12.11.25.
//

#ifndef PHILOXSTATE_H
#define PHILOXSTATE_H
#include <cstdint>

namespace alpaka::rand::internal::engine
{
    /** unified Philox state for single and vector value engine
     *
     * @tparam TCounter Type of the Counter array
     * @tparam TKey Type of the Key array
     */
    template<typename TCounter, typename TKey>
    struct PhiloxState
    {
        using Counter = TCounter;
        using Key = TKey;

        /// Counter array
        Counter counter;
        /// Key array
        Key key;
        /// Intermediate result array
        [[maybe_unused]] Counter result;
        /// Pointer to the active intermediate result element
        [[maybe_unused]] std::uint32_t position;
        // TODO: Box-Muller states
    };
} // namespace alpaka::rand::internal::engine
#endif // PHILOXSTATE_H
