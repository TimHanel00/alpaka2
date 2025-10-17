//
// Created by tim on 17.10.25.
//

#ifndef FRAMESPECTUNINGMODEL_H
#define FRAMESPECTUNINGMODEL_H
#include <alpaka/tune/concepts.hpp>

namespace alpaka::tune
{
    /**
     * @brief Builder-style wrapper that enables tuning of a FrameSpec's kernel-launch arguments.
     *
     * The model wraps a @ref T_FrameSpec (from Alpaka 3) and optionally attaches tuneables
     * for its launch parameters:
     *   - number of frames (`numFrames`)
     *   - frame extent (number of frame elements, `frameExtent`)
     *   - number of blocks (`numBlocks`) -> defined in alpaka::onHost::ThreadSpec
     *   - number of threads (`numThreads`) -> defined in alpaka::onHost::ThreadSpec
     *
     * A tuneable can be a @c alpaka::tune::Tunable (scalar) or @c alpaka::tune::TunableMD (multi-dimensional),
     *
     * If a builder method (e.g. withNumThreadsTune() ) is called without an explicit tuneable,
     * a shallow dummy tuneable is used to **enable** tuning for that dimension while allowing
     * the tuner to choose an appropriate search space.
     *
     * @note This class is the bridge to let the tuner control kernel launch arguments through FrameSpec.
     * @note To enable tuning of individual parameters, chain builder methods:
     * @code
     * using namepace alpaka::tune;
     * auto spec=onHost::FrameSpec{...};
     * auto tune=Tunable{...};
     * auto tunedFrameSpec = FrameSpecTuningModel{spec}
     *                  .withNumFramesTune(tune)
     *                  .withNumThreadsTune();
     * @endcode
     * @tparam T_FrameSpec  Alpaka FrameSpec type (e.g. @c onHost::FrameSpec<...>).
     * @tparam T_NumFramesTune    Tuneable type for number of frames, or @c detail::NoTune.
     * @tparam T_FrameExtentTune  Tuneable type for frame extent, or @c detail::NoTune.
     * @tparam T_NumBlocksTune    Tuneable type for number of blocks, or @c detail::NoTune.
     * @tparam T_NumThreadsTune   Tuneable type for number of threads, or @c detail::NoTune.
     */
    template<
        typename T_FrameSpec
        = onHost::FrameSpec<alpaka::Vec<uint32_t, 1u>, alpaka::Vec<uint32_t, 1u>, alpaka::Vec<uint32_t, 1u>>,
        concepts::TuneableLike T_NumFramesTune = detail::NoTune,
        concepts::TuneableLike T_FrameExtentTune = detail::NoTune,
        concepts::TuneableLike T_NumBlocksTune = detail::NoTune,
        concepts::TuneableLike T_NumThreadsTune = detail::NoTune>
    struct FrameSpecTuningModel
    {
    public:
        /**
         * @brief Construct a tuning model from a FrameSpec and optional tuneables.
         *
         * Any tuneable left as @c detail::NoTune disables tuning for that dimension.
         *
         * @param spec            The underlying FrameSpec to be tuned. Type requires @c
         * alpaka::onHost::isFrameSpec_v<T_FrameSpec>
         * @param numFramesTune   Tuneable controlling the number of frames.
         * @param frameExtentTune Tuneable controlling the frame extent (elements per frame).
         * @param numBlocksTune   Tuneable controlling the number of blocks (ThreadSpec).
         * @param numThreadsTune  Tuneable controlling the number of threads (ThreadSpec).
         *
         * @note Requires @c alpaka::onHost::isFrameSpec_v<T_FrameSpec>.
         */
        constexpr explicit FrameSpecTuningModel(
            T_FrameSpec spec = {},
            T_NumFramesTune numFramesTune = {},
            T_FrameExtentTune frameExtentTune = {},
            T_NumBlocksTune numBlocksTune = {},
            T_NumThreadsTune numThreadsTune = {}) requires(alpaka::onHost::isFrameSpec_v<T_FrameSpec>)
            : m_spec(std::move(spec))
            , m_numFramesTune(std::move(numFramesTune))
            , m_frameExtentTune(std::move(frameExtentTune))
            , m_numBlocksTune(std::move(numBlocksTune))
            , m_numThreadsTune(std::move(numThreadsTune))
        {
        }

        // ------------------------------------------------------------------------
        //  Builder-like methods (all rvalue-qualified for move chaining)
        // ------------------------------------------------------------------------

        /**
         * @brief Enable or replace the tuneable for the number of frames.
         *
         * If called without an argument, a shallow dummy tuneable is injected so the tuner
         * can decide the search space.
         *
         * @tparam T_NumFramesTuneNeu New tuneable type (defaults to a shallow dummy tagged @c tune::frame::numFrames).
         * @param numFramesTune       The tuneable instance to use (optional).
         * @return A new @c FrameSpecTuningModel with @c numFramesTune set.
         *
         * @note When a runtime tuneable is provided, compile-time checks ensure:
         *  - dimensionality matches @c T_FrameSpec::dim()
         *  - value_type matches @c T_FrameSpec::type
         *  - tag is @c tune::frame::numFrames
         *
         * @warning This is rvalue-qualified: call with std::move(model).withNumFramesTune(...).
         */
        template<concepts::TuneableLike T_NumFramesTuneNeu = detail::ShallowTunableDummy<tune::frame::numFrames>>
        constexpr auto withNumFramesTune(T_NumFramesTuneNeu numFramesTune = {}) &&
        {
            if constexpr(concepts::runtimeTuneable<T_NumFramesTuneNeu>)
            {
                static_assert(alpaka::concepts::Vector<typename T_NumFramesTuneNeu::value_type>);
                static_assert(
                    T_FrameSpec::dim() == T_NumFramesTuneNeu::dim,
                    "[withNumFramesTune] Dimension of numFramesTune has to match frameSpec! ");
                static_assert(
                    std::is_same_v<typename T_FrameSpec::type, typename T_NumFramesTuneNeu::value_type::type>,
                    "[withNumFramesTune] Value type of numFramesTune has to match frameSpec! ");
                static_assert(
                    T_NumFramesTuneNeu::tag == tune::frame::numFrames,
                    "[withNumFramesTune] The provided tunable ID does not match the expected. Use "
                    "Tunable<alpaka::tune::numFrame,type>{...} "
                    "or TunableMD<alpaka::tune::numFrame,type>{...}! ");
            }

            return FrameSpecTuningModel<
                T_FrameSpec,
                T_NumFramesTuneNeu,
                T_FrameExtentTune,
                T_NumBlocksTune,
                T_NumThreadsTune>(
                std::move(m_spec),
                std::move(numFramesTune),
                std::move(m_frameExtentTune),
                std::move(m_numBlocksTune),
                std::move(m_numThreadsTune));
        }

        /**
         * @brief Enable or replace the tuneable for frame extent (elements per frame).
         *
         * If called without an argument, a shallow dummy tuneable is injected so the tuner
         * can decide the search space.
         *
         * @tparam T_FrameExtentTuneNeu New tuneable type (defaults to shallow dummy tagged @c
         * tune::frame::frameExtent).
         * @param frameExtentTune       The tuneable instance to use (optional).
         * @return A new @c FrameSpecTuningModel with @c frameExtentTune set.
         *
         * @note When a runtime tuneable is provided, compile-time checks ensure:
         *  - dimensionality matches @c T_FrameSpec::dim()
         *  - value_type matches @c T_FrameSpec::type
         *  - tag is @c tune::frame::frameExtent
         *
         * @warning This is rvalue-qualified: call with std::move(model).withFrameExtentTune(...).
         */
        template<concepts::TuneableLike T_FrameExtentTuneNeu = detail::ShallowTunableDummy<tune::frame::frameExtent>>
        constexpr auto withFrameExtentTune(T_FrameExtentTuneNeu frameExtentTune = {}) &&
        {
            if constexpr(concepts::runtimeTuneable<T_FrameExtentTuneNeu>)
            {
                static_assert(alpaka::concepts::Vector<typename T_FrameExtentTuneNeu::value_type>);
                static_assert(
                    T_FrameSpec::dim() == T_FrameExtentTuneNeu::dim,
                    "[withFrameExtentTune] Dimension of frameExtentTune has to match frameSpec! ");
                static_assert(
                    std::is_same_v<typename T_FrameSpec::type, typename T_FrameExtentTuneNeu::value_type::type>,
                    "[withFrameExtentTune] vector type of frameExtentTune has to match frameSpec! ");
                static_assert(
                    T_FrameExtentTuneNeu::tag == tune::frame::frameExtent,
                    "[withFrameExtentTune] The provided tunable ID does not match the expected. Use "
                    "Tunable<alpaka::tune::frame::frameExtent,type>{...} "
                    "or TunableMD<alpaka::tune::frame::frameExtent,type>{...}! ");
            }
            return FrameSpecTuningModel<
                T_FrameSpec,
                T_NumFramesTune,
                T_FrameExtentTuneNeu,
                T_NumBlocksTune,
                T_NumThreadsTune>(
                std::move(m_spec),
                std::move(m_numFramesTune),
                std::move(frameExtentTune),
                std::move(m_numBlocksTune),
                std::move(m_numThreadsTune));
        }

        /**
         * @brief Enable or replace the tuneable for number of blocks (thread specification).
         *
         * If called without an argument, a shallow dummy tuneable is injected so the tuner
         * can decide the search space.
         *
         * @tparam T_NumBlocksTuneNeu New tuneable type (defaults to shallow dummy tagged @c tune::frame::numBlocks).
         * @param numBlocksTune       The tuneable instance to use (optional).
         * @return A new @c FrameSpecTuningModel with @c numBlocksTune set.
         *
         * @note When a runtime tuneable is provided, compile-time checks ensure:
         *  - dimensionality matches @c T_FrameSpec::dim()
         *  - value_type matches @c T_FrameSpec::type
         *  - tag is @c tune::frame::numBlocks
         *
         * @warning This is rvalue-qualified: call with std::move(model).withNumBlocksTune(...).
         */
        template<concepts::TuneableLike T_NumBlocksTuneNeu = detail::ShallowTunableDummy<tune::frame::numBlocks>>
        constexpr auto withNumBlocksTune(T_NumBlocksTuneNeu numBlocksTune = {}) &&
        {
            if constexpr(concepts::runtimeTuneable<T_NumBlocksTuneNeu>)
            {
                static_assert(alpaka::concepts::Vector<typename T_NumBlocksTuneNeu::value_type>);
                static_assert(
                    T_FrameSpec::dim() == T_NumBlocksTuneNeu::dim,
                    "[withNumBlocksTune] Dimension of numBlocksTune has to match frameSpec! ");
                static_assert(
                    std::is_same_v<typename T_FrameSpec::type, typename T_NumBlocksTuneNeu::value_type::type>,
                    "[withNumBlocksTune] Value type of numBlocksTune has to match frameSpec! ");
                static_assert(
                    T_NumBlocksTuneNeu::tag == tune::frame::numBlocks,
                    "[withNumBlocksTune] The provided tunable ID does not match the expected. Use "
                    "Tunable<alpaka::tune::frame::numBlocks,type>{...} "
                    "or TunableMD<alpaka::tune::frame::numBlocks,type>{...}! ");
            }
            return FrameSpecTuningModel<
                T_FrameSpec,
                T_NumFramesTune,
                T_FrameExtentTune,
                T_NumBlocksTuneNeu,
                T_NumThreadsTune>(
                std::move(m_spec),
                std::move(m_numFramesTune),
                std::move(m_frameExtentTune),
                std::move(numBlocksTune),
                std::move(m_numThreadsTune));
        }

        /**
         * @brief Enable or replace the tuneable for number of threads (thread specification).
         *
         * If called without an argument, a shallow dummy tuneable is injected so the tuner
         * can decide the search space.
         *
         * @tparam T_NumThreadsTuneNeu New tuneable type (defaults to shallow dummy tagged @c tune::frame::numThreads).
         * @param numThreadsTune       The tuneable instance to use (optional).
         * @return A new @c FrameSpecTuningModel with @c numThreadsTune set.
         *
         * @note When a runtime tuneable is provided, compile-time checks ensure:
         *  - dimensionality matches @c T_FrameSpec::dim()
         *  - value_type matches @c T_FrameSpec::type
         *  - tag is @c tune::frame::numThreads
         *
         * @warning This is rvalue-qualified: call with std::move(model).withNumThreadsTune(...).
         */
        template<concepts::TuneableLike T_NumThreadsTuneNeu = detail::ShallowTunableDummy<tune::frame::numThreads>>
        constexpr auto withNumThreadsTune(T_NumThreadsTuneNeu numThreadsTune = {}) &&
        {
            if constexpr(concepts::runtimeTuneable<T_NumThreadsTuneNeu>)
            {
                static_assert(alpaka::concepts::Vector<typename T_NumThreadsTuneNeu::value_type>);
                static_assert(
                    T_FrameSpec::dim() == T_NumThreadsTuneNeu::dim,
                    "[withNumThreadsTune] Dimension of numBlocksTune has to match frameSpec! ");
                static_assert(
                    std::is_same_v<typename T_FrameSpec::type, typename T_NumThreadsTuneNeu::value_type::type>,
                    "[withNumThreadsTune] Value type of numBlocksTune has to match frameSpec! ");
                static_assert(
                    T_NumThreadsTuneNeu::tag == tune::frame::numThreads,
                    "[withNumThreadsTune] The provided tunable ID does not match the expected. Use "
                    "Tunable<alpaka::tune::frame::numThreads,type>{...} "
                    "or TunableMD<alpaka::tune::frame::numThreads,type>{...}! ");
            }
            return FrameSpecTuningModel<
                T_FrameSpec,
                T_NumFramesTune,
                T_FrameExtentTune,
                T_NumBlocksTune,
                T_NumThreadsTuneNeu>(
                std::move(m_spec),
                std::move(m_numFramesTune),
                std::move(m_frameExtentTune),
                std::move(m_numBlocksTune),
                std::move(numThreadsTune));
        }

        // ------------------------------------------------------------------------
        //  Query helpers
        // ------------------------------------------------------------------------

        /**
         * @brief Check whether a tuneable for number of frames is present.
         * @return true if @c T_NumFramesTune is not @c detail::NoTune.
         */
        [[nodiscard]] constexpr bool hasNumFramesTune() const noexcept
        {
            return !std::is_same_v<T_NumFramesTune, alpaka::tune::detail::NoTune>;
        }

        /**
         * @brief Check whether a tuneable for frame extent is present.
         * @return true if @c T_FrameExtentTune is not @c detail::NoTune.
         */
        [[nodiscard]] constexpr bool hasFrameExtentTune() const noexcept
        {
            return !std::is_same_v<T_FrameExtentTune, alpaka::tune::detail::NoTune>;
        }

        /**
         * @brief Check whether a tuneable for number of blocks is present.
         * @return true if @c T_NumBlocksTune is not @c detail::NoTune.
         */
        [[nodiscard]] constexpr bool hasNumBlocksTune() const noexcept
        {
            return !std::is_same_v<T_NumBlocksTune, alpaka::tune::detail::NoTune>;
        }

        /**
         * @brief Check whether a tuneable for number of threads is present.
         * @return true if @c T_NumThreadsTune is not @c detail::NoTune.
         */
        [[nodiscard]] constexpr bool hasNumThreadsTune() const noexcept
        {
            return !std::is_same_v<T_NumThreadsTune, alpaka::tune::detail::NoTune>;
        }

        /**
         * @brief Access the tuneable for number of frames.
         * @return The stored tuneable object.
         * @pre @c hasNumFramesTune() must be true.
         */
        [[nodiscard]] constexpr auto getNumFramesTune() const noexcept
        {
            static_assert(
                !std::is_same_v<T_NumFramesTune, alpaka::tune::detail::NoTune>,
                "FrameSpecTuningModel has no NumFramesTune!");
            return m_numFramesTune;
        }

        /**
         * @brief Access the tuneable for frame extent.
         * @return The stored tuneable object.
         * @pre @c hasFrameExtentTune() must be true.
         */
        [[nodiscard]] constexpr auto getFrameExtentTune() const noexcept
        {
            static_assert(
                !std::is_same_v<T_FrameExtentTune, alpaka::tune::detail::NoTune>,
                "FrameSpecTuningModel has no frameExtentTune!");
            return m_frameExtentTune;
        }

        /**
         * @brief Access the tuneable for number of blocks.
         * @return The stored tuneable object.
         * @pre @c hasNumBlocksTune() must be true.
         */
        [[nodiscard]] constexpr auto getNumBlocksTune() const noexcept
        {
            static_assert(
                !std::is_same_v<T_NumBlocksTune, alpaka::tune::detail::NoTune>,
                "FrameSpecTuningModel has no NumBlocksTune!");
            return m_numBlocksTune;
        }

        /**
         * @brief Access the tuneable for number of threads.
         * @return The stored tuneable object.
         * @pre @c hasNumThreadsTune() must be true.
         */
        [[nodiscard]] constexpr auto getNumThreadsTune() const noexcept
        {
            static_assert(
                !std::is_same_v<T_NumThreadsTune, alpaka::tune::detail::NoTune>,
                "FrameSpecTuningModel has no NumThreadsTune!");
            return m_numThreadsTune;
        }

    private:
        // ------------------------------------------------------------------------
        //  Data Members
        // ------------------------------------------------------------------------

        /** @brief Tuneable for number of frames, or @c detail::NoTune. */
        T_NumFramesTune m_numFramesTune;

        /** @brief Tuneable for frame extent (elements per frame), or @c detail::NoTune. */
        T_FrameExtentTune m_frameExtentTune;

        /** @brief Tuneable for number of blocks (thread specification), or @c detail::NoTune. */
        T_NumBlocksTune m_numBlocksTune;

        /** @brief Tuneable for number of threads (thread specification), or @c detail::NoTune. */
        T_NumThreadsTune m_numThreadsTune;

        /** @brief The underlying FrameSpec being tuned. */
        T_FrameSpec m_spec;
    };


} // namespace alpaka::tune
#endif // FRAMESPECTUNINGMODEL_H
