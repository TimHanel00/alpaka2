//
// Created by tim on 12.11.25.
//

#ifndef UNIFORMREAL_H
#define UNIFORMREAL_H
#include <alpaka/core/common.hpp>
#include <alpaka/rand/engines/philox/philox.hpp>

namespace alpaka::rand::internal::concepts
{
    template<typename TResult>
    concept Floating = std::is_floating_point_v<TResult>;
    template<typename TResult>
    concept Integer = std::is_integral_v<TResult>;
} // namespace alpaka::rand::internal::concepts

namespace alpaka::rand::distribution
{
    // set the interval range for a distribution
    enum Interval
    {
        //(a,b) (excludes a and b)
        OO,
        //[a,b) (includes a and excludes b)
        CO,
        //(a,b] (excludes a and includes b)
        OC,
        //[a,b] (includes a and b)
        CC
    };
} // namespace alpaka::rand::distribution

namespace alpaka::rand::internal
{
    template<typename T>
    struct Dummy;
    template<distribution::Interval K>
    struct DummyI;

    /** Convert an integer RNG result to a floating-point value.
     *
     * This is the fallback implementation used when no interval specialization
     * matches. It should never be instantiated and exists only to
     * catch unsupported interval configurations.
     */
    template<typename T_Engine, distribution::Interval Interval_, typename T_Integer, typename T_Floating>
    struct IntervalAwareConversion
    {
        DummyI<Interval_> n_type;
        Dummy<T_Integer> i_type;
        Dummy<T_Floating> r_type;

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const&) const {

        };
    };

    /** Converts an integer RNG output to a floating point type in the interval [0, 1). */
    template<typename T_Engine, typename T_Integer, typename T_Floating>
    struct IntervalAwareConversion<T_Engine, distribution::CO, T_Integer, T_Floating>
    {
        static constexpr T_Floating val
            = static_cast<T_Floating>(std::numeric_limits<T_Integer>::max()) + static_cast<T_Floating>(1);

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) / val;
        };
    };

    /** Convert an integer RNG output to a floating point type in the interval (0, 1].
     * @note: implemented by mirroring the result of [0,1) (using 1-x) -> might add unintended performance overhead
     */
    template<typename T_Engine, typename T_Integer, typename T_Floating>
    struct IntervalAwareConversion<T_Engine, distribution::OC, T_Integer, T_Floating>
    {
        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            // mirror distribution
            return static_cast<T_Floating>(1)
                   - IntervalAwareConversion<T_Engine, distribution::CO, T_Integer, T_Floating>{}(i);
        };
    };

    /** Convert an integer RNG output to a floating point type in the closed interval [0, 1].
     */
    template<typename T_Engine, typename T_Integer, typename T_Floating>
    struct IntervalAwareConversion<T_Engine, distribution::CC, T_Integer, T_Floating>
    {
        static constexpr T_Floating val = static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) / val;
        };
    };

    /** Adapt the bit length of the engine output to match the target type.
     * This is the default case where the engine result type already matches and thus the engine is simply invoked.
     */
    template<typename T_Engine, uint32_t byteLengthEngineResult, uint32_t byteLengthRealType>
    struct bitLengthConformityAdapter
    {
        static_assert(
            (byteLengthEngineResult == 4u || byteLengthRealType == 8u),
            "Result returned by the randomBitGenerator does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            (byteLengthEngineResult == 4u || byteLengthRealType == 8u),
            "The requested floating point type does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            byteLengthEngineResult == byteLengthRealType,
            "By logic this should never fail in case the compiler accepts the specialization of the adapter!");

        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return engine();
        }
    };

    /** Adapts a 32-bit engine output to a 64-bit value. This involves invoking the engine twice. */
    template<typename T_Engine>
    struct bitLengthConformityAdapter<T_Engine, 4u, 8u>
    {
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return static_cast<uint64_t>(engine()) << 32 | static_cast<uint64_t>(engine());
        }
    };

    /** Adapt a 64-bit engine output to a 32-bit value. Uses a simple narrowing conversion.*/
    template<typename T_Engine>
    struct bitLengthConformityAdapter<T_Engine, 8u, 4u>
    {
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return static_cast<uint32_t>(engine());
        }
    };

    /** Generate a floating-point value in the requested interval.
     *
     * Adapts the engine output to the required bit length, converts the integer
     * to a normalized floating point value in the resquested interval, and performs
     * resampling when needed (e.g. for the open interval (0,1)).*/
    template<distribution::Interval Interval_value, typename T_Engine, typename T_Result>
    ALPAKA_FN_HOST_ACC auto randomRealDispatch(T_Engine& engine) -> T_Result
    {
        using T_EngineResult = std::remove_cvref_t<decltype(engine())>;
        // generates an integer the length of the size T_Result
        auto adaptedBits = bitLengthConformityAdapter<
            T_Engine,
            static_cast<uint32_t>(sizeof(T_EngineResult)),
            static_cast<uint32_t>(sizeof(T_Result))>{}(engine);
        // the two sided open interval represents a special case where we simply resample in case we hit a one
        if constexpr(Interval_value == distribution::OO)
        {
            T_Result res;
            do
            {
                adaptedBits = bitLengthConformityAdapter<T_Engine, sizeof(T_EngineResult), sizeof(T_Result)>{}(engine);
                res = IntervalAwareConversion<T_Engine, distribution::OC, ALPAKA_TYPEOF(adaptedBits), T_Result>{}(
                    adaptedBits);
            } while(res >= T_Result(1.0));

            return res;
        }
        else
        {
            return IntervalAwareConversion<T_Engine, Interval_value, ALPAKA_TYPEOF(adaptedBits), T_Result>{}(
                adaptedBits);
        }
    }
    template<typename T_Engine, uint32_t TResultSize, uint32_t TElemSize, uint32_t TElems>
    struct vectorDispatchWrapper;

    template<typename T_Engine, uint32_t TElemSize, uint32_t TElems>
    struct vectorDispatchWrapper<T_Engine, 4u, TElemSize, TElems>
    {
        T_Engine& ph;
        static_assert(TElems > 0, "RandomEngine did not return any elements!");

        ALPAKA_FN_HOST_ACC explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        ALPAKA_FN_HOST_ACC uint32_t operator()() const
        {
            auto res = ph();
            return static_cast<uint32_t>(res[0]);
        }
    };

    // wrapper specialization to let a engine::Philox4x32x10Vector or
    // similar engines implement uniformReal<double> more efficiently (without needing to call engine() twice)
    template<typename T_Engine, uint32_t TElems>
    struct vectorDispatchWrapper<T_Engine, 8u, 4u, TElems>
    {
        T_Engine& ph;
        using TResult = decltype(ph());
        static constexpr auto dim = TResult::dim();
        static_assert(TElems >= 2, "Engine result dimension must be >= 2, to be usable in UniformReal<double>");

        ALPAKA_FN_HOST_ACC explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        ALPAKA_FN_HOST_ACC uint64_t operator()() const
        {
            auto res = ph();
            return (static_cast<uint64_t>(res[0]) << 32) | static_cast<uint64_t>(res[1]);
        }
    };

    template<alpaka::rand::internal::concepts::Floating T, distribution::Interval Interval_v>
    class UniformRealBase
    {
    public:
        using value_type = T;

        ALPAKA_FN_HOST_ACC constexpr UniformRealBase() : UniformRealBase(0, 1)
        {
        }

        ALPAKA_FN_HOST_ACC constexpr UniformRealBase(T min, T max) : _min(min), _max(max), _range(_max - _min)
        {
        }

    protected:
        T const _min;
        T const _max;
        T const _range;
    };
} // namespace alpaka::rand::internal

namespace alpaka::rand::distribution
{
    /** Select a floating-point value from a uniform interval.
     *
     * This generator produces floating-point values of type `T_Result` drawn from a uniform
     * interval `[a, b)` or `(a, b]`, depending on the interval type specified via `Interval_v`: default case is CO
     * ->[a,b). The interface mirrors `std::uniform_real_distribution`, and can be invoked with any engine fulfilling
     * the `std::uniform_random_bit_generator` concept.
     *
     * The distribution currently supports single- and double precision floating-point result types and
     * random engines returning 32 or 64 unsigned integer types.
     */
    template<alpaka::rand::internal::concepts::Floating T_Result, Interval Interval_v = CO>
    class UniformReal : internal::UniformRealBase<T_Result, Interval_v>
    {
        using Base = internal::UniformRealBase<T_Result, Interval_v>;
        using Base::Base;

    public:
        template<std::uniform_random_bit_generator T_Engine>
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine) -> T_Result
        {
            T_Result res = internal::randomRealDispatch<Interval_v, T_Engine, T_Result>(engine);
            // @TODO potentially add underflow protection as suggested by https://doi.org/10.1145/3503512
            return res * this->_range + this->_min;
        }

        ALPAKA_FN_HOST_ACC auto operator()(engine::Philox4x32x10Vector& engine)
        {
            using T_EngineResult = ALPAKA_TYPEOF(engine());
            // internal::Dummy<T_EngineResult> dummy2;
            using valueType = ALPAKA_TYPEOF(engine()[0]);
            static constexpr auto dim = T_EngineResult::dim();
            auto dispatchWrapper = internal::vectorDispatchWrapper<
                engine::Philox4x32x10Vector,
                static_cast<uint32_t>(sizeof(T_Result)),
                static_cast<uint32_t>(sizeof(valueType)),
                dim>(engine);
            using TdispatchWrapper = decltype(dispatchWrapper);
            T_Result res = internal::randomRealDispatch<Interval_v, TdispatchWrapper, T_Result>(dispatchWrapper);
            return res * this->_range + this->_min;
        }
    };
} // namespace alpaka::rand::distribution
#endif // UNIFORMREAL_H
