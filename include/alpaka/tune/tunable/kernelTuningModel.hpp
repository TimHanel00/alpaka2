//
// Created by tim on 06.04.25.
//

#ifndef KernelTuningModel_H
#define KernelTuningModel_H
#include <alpaka/tune/IO/config.hpp>
#include <alpaka/tune/tunable/Tunable.hpp>
#include <alpaka/tune/utils/tupleHelper.hpp>

#include <cmath>
#include <cstddef>

namespace alpaka::tune
{
    namespace detail
    {
        template<typename Tuple, std::size_t... Is>
        constexpr auto makeOffsetsImpl(std::index_sequence<Is...>)
        {
            constexpr std::array<std::size_t, sizeof...(Is)> arr = []
            {
                std::array<std::size_t, sizeof...(Is)> a{};
                std::size_t acc = 0;
                std::size_t i = 0;
                ((a[i++] = acc, acc += std::tuple_element_t<Is, Tuple>::dim), ...);
                return a;
            }();
            return arr;
        }

        template<typename Tuple>
        constexpr auto makeOffsets()
        {
            return makeOffsetsImpl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
        }
    } // namespace detail

    /**
     * @brief A ParameterAccessor provides a view on the concrete value of a tunable parameter.
     * It also contains information about the name, ID, and kind of the corresponding tunable object.
     *
     * @tparam T_Value     The underlying value type of the parameter.
     * @tparam TuneableId  A compile-time constant identifier associated with the tunable parameter.
     * @tparam TuneableKind The kind of tunable parameter (e.g.,Tuneable,TuneableMD,CTuneable),
     *                      represented by alpaka::tune::TuneableKind.
     *
     * @details
     * Each ParameterAccessor instance acts as a lightweight view on an existing tunable configuration value.
     * It does not own the value, but holds a constant reference to it together with a reference to its name.
     */
    template<typename T_Value, uint32_t TuneableId, alpaka::tune::TunableKind TuneableKind>
    struct ParameterAccessor
    {
        static constexpr uint32_t ID = TuneableId;
        static constexpr alpaka::tune::TunableKind kind = TuneableKind;
        T_Value m_value;
        std::string const& m_name;
    };

    template<typename T_KernelTuningModel>
    struct ConfigDescriptor
    {
        explicit ConfigDescriptor(T_KernelTuningModel const& tuningModel)
            : m_kernelTuningModel(m_kernelTuningModel) {};

        /**
         * @brief Construct a tuple of ParameterAccessors from a parameter configuration (Config).
         * A parameterAccessor contains the Values of the tuneable parameter corresponding to this Config alongside
         * metadata (name, ID, kind). For Ctuneables the value is not returned directly.
         *
         * @param config The parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * deducible).
         */
        template<typename idx_type, auto NumTuneables>
        constexpr auto getValuesFromConfig(config::Config<idx_type, NumTuneables> const& config)
        {
            m_kernelTuningModel.getValuesFromConfig(config);
        }

        /**
         * @brief turns a normalized Configuration (floating based) into a integer based configuration*/
        template<alpaka::tune::concepts::Floating floating_type, auto NumTuneables>
        constexpr auto createConfigFromNormalized(
            config::NormalizedConfig<floating_type, NumTuneables> const& config) const
        {
            return m_kernelTuningModel.createConfigFromNormalized(config);
        }

        /**
         *
         * @returns a default initialized configuration
         */
        static constexpr auto getEmptyConfig()
        {
            return config::Config<uint32_t, T_KernelTuningModel::numDims>{};
        }

        /**
         *
         * @returns a default initialized normalized configuration
         */
        static constexpr auto getEmptyNormalizedConfig()
        {
            return config::NormalizedConfig<double_t, T_KernelTuningModel::numDims>{};
        }

        /**
         * @brief get the number of values for each dimension in the parameter space
         *
         * @return std::array<uint32_t,numDims>
         *
         */
        auto getNumValues() const
        {
            return m_kernelTuningModel.m_numValues;
        }

    private:
        T_KernelTuningModel const& m_kernelTuningModel;
    };

    template<typename T_FrameTuple, typename T_UserTuple, typename T_CompileTuple>
    auto makeAllTuneables(T_FrameTuple& frame_tuple, T_UserTuple& user_tuple, T_CompileTuple& compile_tuple)
    {
        return std::apply(
            [&](auto&... compileElems)
            {
                return std::apply(
                    [&](auto&... userElems)
                    {
                        return std::apply(
                            [&](auto&... frameElems)
                            { return std::tie(frameElems..., userElems..., compileElems...); },
                            frame_tuple);
                    },
                    user_tuple);
            },
            compile_tuple);
    }

    template<
        typename T_FrameTuneables = std::tuple<>,
        typename T_UserTuple = std::tuple<>,
        typename T_CompileTimeTuple = std::tuple<>>
    struct KernelTuningModel
    {
        T_FrameTuneables m_frameTuneables;
        T_UserTuple m_userTuneables;
        T_CompileTimeTuple m_compileTimeTuneables;


        using T_allTuneablesBare = utils::tuple_cat_t<T_FrameTuneables, T_UserTuple, T_CompileTimeTuple>;
        using T_allTuneablesRef
            = decltype(makeAllTuneables(m_frameTuneables, m_userTuneables, m_compileTimeTuneables));
        T_allTuneablesRef m_allTuneables;
        static constexpr auto numTunables = std::tuple_size_v<T_allTuneablesBare>;

        static constexpr auto numDims = []
        {
            std::size_t acc = 0;
            [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                ((acc += std::tuple_element_t<Is, T_allTuneablesBare>::dim), ...);
            }(std::make_index_sequence<numTunables>{});
            return acc;
        }();
        std::array<uint32_t, numDims> m_numValues{};
        constexpr KernelTuningModel() = default;

        constexpr KernelTuningModel(T_FrameTuneables frameT, T_UserTuple userT, T_CompileTimeTuple compileT)
            : m_frameTuneables(std::move(frameT))
            , m_userTuneables(std::move(userT))
            , m_compileTimeTuneables(std::move(compileT))
            , m_allTuneables(makeAllTuneables(m_frameTuneables, m_userTuneables, m_compileTimeTuneables))
            , m_numValues(getNumValues())
        {
        }

        /**
         * @brief turns a normalized Configuration (floating based) into a integer based configuration*/
        template<alpaka::tune::concepts::Floating floating_type, auto NumTuneables>
        constexpr auto createConfigFromNormalized(
            config::NormalizedConfig<floating_type, NumTuneables> const& config) const
        {
            static_assert(
                NumTuneables == numDims,
                " Normalized Config does not have the correct number of parameters! ");
            std::array<std::uint32_t, NumTuneables> return_config_ar;
            for(auto val = 0; val < NumTuneables; val++)
            {
                // floor the result of the operation
                return_config_ar[val] = static_cast<std::uint32_t>(return_config_ar[val] * m_numValues[val]);
                if(return_config_ar[val] == m_numValues[val])
                {
                    // prevent exceeding max number of config arguments in subsequent calls
                    auto tmp = return_config_ar[val] - 1;
                    // check underflow
                    return_config_ar[val] = tmp < return_config_ar[val] ? tmp : return_config_ar[val];
                }
            }
            return config::Config{return_config_ar};
        }

        /**
         * @brief returns the indicies corresponding to the compile-time tuneable indicies of the parameter
         *configuration as a std::array<idx_type,std::tuple_size_v<T_CompileTimeTuple>>
         *
         * @param config The parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * @return std::array<idx_type,std::tuple_size_v<T_CompileTimeTuple>>
         **/
        template<typename idx_type, auto NumTuneables>
        [[nodiscard]] constexpr std::array<idx_type, std::tuple_size_v<T_CompileTimeTuple>>
        getConfigSubset_CompileTuneables(config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = std::tuple_size_v<T_UserTuple> + std::tuple_size_v<T_FrameTuneables>;
            constexpr auto offsets = detail::makeOffsets<T_allTuneablesBare>();
            std::array<idx_type, std::tuple_size_v<T_CompileTimeTuple>> ret;
            for(std::size_t i = offsets[startingIdx], a = 0; i < NumTuneables, a < ret.size(); ++i, a++)
            {
                ret[a] = config[i];
            }
            return ret;
        }

        /**
         * @brief Construct a tuple of ParameterAccessors from a parameter configuration (Config).
         * A parameterAccessor contains the Values of the tuneable parameter corresponding to this Config alongside
         * metadata (m_name, ID, kind) as a std::tuple.
         * More concretely member m_value contains the values of the produced configuration for Tuneable and
         * TuneableMD. The values for the compile-time tuneable have can be accessed using std::visit:
         * std::visit([](auto&& v) {  }, std::get<I>(parameterAccessor).m_value);
         *
         * @param config The parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * deducible).
         */
        template<typename idx_type, auto NumTuneables>
        constexpr auto getValuesFromConfig(config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = 0;
            return getValuesFromTuple<startingIdx, idx_type, NumTuneables, T_allTuneablesRef>(config, m_allTuneables);
        }

        auto getNumValues() const
        {
            std::array<uint32_t, numDims> numValues;
            uint32_t curIdx = 0;
            alpaka::tune::utils::for_each(
                m_allTuneables,
                [&](auto const& elem)
                {
                    auto vec = elem.getNumValues(); // this is runtime
                    for(auto dim = 0; dim < alpaka::getDim(vec) && curIdx < numDims; dim++, ++curIdx)
                    {
                        numValues[curIdx] = vec[dim];
                    }
                });
            return numValues;
        }

        auto getInitConfig()
        {
            auto ar = std::array<uint32_t, numDims>{};
            auto idx = 0;
            alpaka::tune::utils::for_each(
                m_allTuneables,
                [&](auto const& elem)
                {
                    auto startIdxWrapper = elem.startingIndex; // this is runtime
                    if(startIdxWrapper.has_value())
                    {
                        auto startIdxValue = startIdxWrapper.value();
                        if constexpr(alpaka::isVector_v<decltype(startIdxValue)>)
                        {
                            for(auto i = 0; i < alpaka::getDim(startIdxValue); i++)
                            {
                                ar[idx++] = startIdxValue[i];
                            }
                            return;
                        }
                        else
                        {
                            ar[idx++] = startIdxValue;
                            return;
                        }
                    }
                    ar[idx++] = 0;
                });
            return config::Config{ar};
        }

        [[nodiscard]] std::size_t getMaxPossibleRuns() const
        {
            auto vec = getNumValues();
            std::size_t maxPossibleRuns = 1;
            for(uint32_t i = 0; i < alpaka::getDim(vec); i++)
            {
                auto val = static_cast<std::size_t>(vec[i]);
                if(val == 0)
                {
                    return 0;
                }
                auto tmp = maxPossibleRuns * vec[i];
                if(tmp < maxPossibleRuns)
                {
                    std::cerr << " overflow during tuning space calculation, this error can be ignored if the tuning "
                                 "space is restricted by environment variables TunerMaxConfigEvaluations or "
                                 "TunerMaxCheckedConfigs! "
                              << std::endl;
                    return std::numeric_limits<std::size_t>::max();
                }
                maxPossibleRuns = tmp;
            }
            return maxPossibleRuns;
        }

        /**
         * @brief Construct a tuple of ParameterAccessors from a parameter configuration (Config).
         * A parameterAccessor contains the Values of the tuneable parameter corresponding to the runtime-Tuneable
         * subset of this Config alongside metadata (m_name, ID, kind) as a std::tuple. More concretely member
         * m_value contains the user defined runtime-tunables values corresponding to this parameter configuration.
         *
         * @param config The (whole) parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * deducible).
         */
        template<typename idx_type, auto NumTuneables>
        [[nodiscard]] constexpr auto getValuesForRuntimeTuneables(
            config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = std::tuple_size_v<T_FrameTuneables>;
            return getValuesFromTuple<startingIdx, idx_type, NumTuneables, T_UserTuple>(config, m_userTuneables);
        }

        /**
         * @brief Construct a tuple of ParameterAccessors from a parameter configuration (Config).
         * A parameterAccessor contains the Values of the tuneable parameter corresponding to the
         * compile-time-Tuneable subset of this Config alongside metadata (m_name, ID, kind) as a std::tuple. More
         * concretely member m_value contains the the compileTime-tunables values corresponding to this parameter
         * configuration as a std::variant. It can be access using std::visit([](auto&& v) {  },
         * std::get<I>(parameterAccessor).m_value); where index: I -> corresponds to the I`th compile-time
         * parameter index.
         *
         * @param config The (whole) parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * deducible).
         */
        template<typename idx_type, auto NumTuneables>
        [[nodiscard]] constexpr auto getValuesForCompileTuneables(
            config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = std::tuple_size_v<T_UserTuple> + std::tuple_size_v<T_FrameTuneables>;
            return getValuesFromTuple<startingIdx, idx_type, NumTuneables, T_CompileTimeTuple>(
                config,
                m_compileTimeTuneables);
        }

        /**
         * @brief Construct a tuple of ParameterAccessors from a parameter configuration (Config).
         * A parameterAccessor contains the Values of the tuneable parameter corresponding to the frameSec-Tuneable
         *  subset of this Config alongside metadata (m_name, ID, kind) as a std::tuple. The member m_value
         * contains the frameSpec-tuneable values corresponding to this parameter configuration.
         *
         * @param config The parameter configuration.
         * @tparam idx_type The idx_type of the parameter configuration (should be automatically deducible).
         * @tparam NumTuneables The number of Tuneables of the parameter configuration (should be automatically
         * deducible).
         */
        template<typename idx_type, auto NumTuneables>
        [[nodiscard]] constexpr auto getValuesForFrameTuneables(
            config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = 0;
            return getValuesFromTuple<startingIdx, idx_type, NumTuneables, T_FrameTuneables>(config, m_frameTuneables);
        }

        /**
         * @brief Applies a parameter configuration to a frame-specification.
         */
        template<
            alpaka::concepts::Vector T_Frames,
            alpaka::concepts::Vector T_Extent,
            alpaka::concepts::Vector T_Blocks,
            typename idx_type,
            auto NumTuneables>
        void applyToFrameSpec(
            alpaka::onHost::FrameSpec<T_Frames, T_Extent, T_Blocks>& frame_spec,
            config::Config<idx_type, NumTuneables> const& config) const
        {
            constexpr std::size_t startingIdx = 0;
            auto frameTunes = getValuesForFrameTuneables(config);
            static constexpr auto size = std::tuple_size_v<decltype(frameTunes)>;
            if constexpr(size != 0)
            {
                alpaka::tune::utils::for_each_enumerate(
                    frameTunes,
                    [&]<std::size_t I, typename T0>(T0 const& tuneable)
                    {
                        if constexpr(T0::ID == alpaka::tune::frame::numBlocks)
                        {
                            frame_spec.m_threadSpec.m_numBlocks = std::get<I>(frameTunes).m_value;
                        }
                        else if constexpr(T0::ID == alpaka::tune::frame::frameExtent)
                        {
                            frame_spec.m_frameExtent = std::get<I>(frameTunes).m_value;
                        }
                        else if constexpr(T0::ID == alpaka::tune::frame::numFrames)
                        {
                            frame_spec.m_numFrames = std::get<I>(frameTunes).m_value;
                        }
                        else if constexpr(T0::ID == alpaka::tune::frame::numThreads)
                        {
                            frame_spec.m_threadSpec.m_numThreads = std::get<I>(frameTunes).m_value;
                        }
                    });
            }
        }

        template<typename Tuple, auto Name, std::size_t I = 0>
        static constexpr bool hasTagInTuple()
        {
            if constexpr(I < std::tuple_size_v<Tuple>)
            {
                using Elem = std::tuple_element_t<I, std::remove_cvref_t<Tuple>>;
                if constexpr(std::is_same_v<Elem, alpaka::tune::detail::NoTune>)
                {
                    return false;
                }
                else
                {
                    return (Elem::tag == Name) || hasTagInTuple<Tuple, Name, I + 1>();
                }
            }
            else
            {
                return false;
            }
        }

        template<typename Tuple, auto ID>
        static constexpr bool hasTuneableTag()
        {
            return hasTagInTuple<Tuple, ID>();
        }

        template<auto ID>
        static constexpr bool hasFrameTuneable()
        {
            constexpr auto ID_v = static_cast<std::size_t>(ID);
            return hasTuneableTag<T_FrameTuneables, ID_v>();
        }

        template<auto ID>
        static constexpr bool hasUserTuneable()
        {
            constexpr auto ID_v = static_cast<std::size_t>(ID);
            return hasTuneableTag<T_UserTuple, ID_v>();
        }

        template<auto ID>
        static constexpr bool hasTuneable()
        {
            return hasTuneableTag<T_allTuneablesBare, ID>();
        }

        static constexpr bool hasNumBlocksTune()
        {
            return hasFrameTuneable<alpaka::tune::detail::SpecialTuneableID::numBlocks>();
        }

        static constexpr bool hasNumThreadsTune()
        {
            return hasFrameTuneable<alpaka::tune::detail::SpecialTuneableID::numThreads>();
        }

        static constexpr bool hasNumFramesTune()
        {
            return hasFrameTuneable<alpaka::tune::detail::SpecialTuneableID::numFrames>();
        }

        static constexpr bool hasFrameExtentTune()
        {
            return hasFrameTuneable<alpaka::tune::detail::SpecialTuneableID::frameExtent>();
        }

        constexpr auto& getNumBlocksTune()
        {
            return *getByID<alpaka::tune::detail::SpecialTuneableID::numBlocks>();
        }

        constexpr auto& getThreadBlockSizeTune()
        {
            return *getByID<alpaka::tune::detail::SpecialTuneableID::numThreads>();
        }

        constexpr auto& getNumFramesTune()
        {
            return *getByID<alpaka::tune::detail::SpecialTuneableID::numFrames>();
        }

        constexpr auto& getFrameExtentTune()
        {
            return *getByID<alpaka::tune::detail::SpecialTuneableID::frameExtent>();
        }

        template<std::size_t I = 0, typename Tuple, auto ID>
        constexpr auto* getByIDImpl(Tuple& tuple) const
        {
            if constexpr(I < std::tuple_size_v<Tuple>)
            {
                auto& elem = std::get<I>(tuple);
                using elemType = std::decay_t<decltype(elem)>;
                if constexpr(elemType::tag == ID)
                {
                    return &elem;
                }
                else
                {
                    return getByIDImpl<I + 1, Tuple, ID>(tuple);
                }
            }
            else
            {
                return &alpaka::tune::detail::noTune; // or nullptr if appropriate
            }
        }

        template<auto... ID>
        constexpr auto getByIDs()
        {
            return std::make_tuple(getByID<ID>()...); // expands each getByID
        }

        template<auto ID>
        constexpr auto* getByID()
        {
            constexpr auto ID_v = static_cast<std::size_t>(ID);
            return getByIDImpl<0, T_allTuneablesRef, ID_v>(m_allTuneables);
        }

    private:
        /**
         * @brief Internal constexpr implementation.
         *
         * @details
         *  Uses compile-time prefix sums of `dim` to map each Tuneable
         *  to the correct section of the configuration array.
         *  Each Tuneable is converted into a ParameterAccessor containing:
         *   - `T_Value`  → The value type (scalar or Vec)
         *   - `TuneableId` → The Tuneable’s compile-time index (I)
         *   - `TuneableKind` → Whether it is CTuneable or RTuneable
         */
        template<auto startingIndex, typename idx_type, auto NumTuneables, typename SubsetTuple>
        constexpr auto getValuesFromTuple(
            alpaka::tune::config::Config<idx_type, NumTuneables> const& config,
            SubsetTuple const& tuneablesSubset) const
        {
            static_assert(NumTuneables == numDims, " Config has the wrong number of parameters!");
            constexpr std::size_t N = std::tuple_size_v<std::remove_reference_t<SubsetTuple>>;

            // compute offsets for all tuneables in the global tuple
            constexpr auto offsets = detail::makeOffsets<T_allTuneablesBare>();


            auto handleTuneable = [&]<typename T0>(T0 const& tuneable, auto I_c)
            {
                using T = std::decay_t<T0>;
                constexpr std::size_t fullIndex = startingIndex + I_c; // correct slice in the global tuple
                constexpr std::size_t offset = offsets[fullIndex];
                constexpr auto kind = T::tuneableType;
                constexpr std::size_t ID = T::tag; // or fullIndex if tag not available

                if constexpr(kind == TunableKind::CTunable)
                {
                    auto valueVariant = alpaka::tune::utils::visitIndexVariant<std::tuple_size_v<typename T0::Values>>(
                        config[offset],
                        tuneable); // variant use std::visit for runtime dispatch

                    return std::tuple{
                        ParameterAccessor<decltype(valueVariant), ID, kind>{valueVariant, tuneable.getName()}};
                }
                else
                {
                    if constexpr(kind == alpaka::tune::TunableKind::TunableMD)
                    {
                        alpaka::Vec<idx_type, T::dim> vec;
                        for(std::size_t j = 0; j < T::dim; ++j)
                            vec[j] = config[offset + j];

                        auto const value = tuneable.getValueByIndex(vec); // call by value
                        return std::tuple{ParameterAccessor<decltype(value), ID, kind>{value, tuneable.getName()}};
                    }
                    else
                    {
                        if constexpr(kind == alpaka::tune::TunableKind::Tunable)
                        {
                            // TunableKind: Tuneable
                            alpaka::Vec<idx_type, T::dim> vec;
                            for(std::size_t j = 0; j < T::dim; ++j)
                                vec[j] = config[offset + j];

                            auto const& value = tuneable.getValueByIndex(vec); // call by reference
                            return std::tuple{ParameterAccessor<decltype(value), ID, kind>{value, tuneable.getName()}};
                        }
                    }
                }
            };
            auto outside_tuple = [&]<std::size_t... I>(std::index_sequence<I...>)
            {
                return std::tuple_cat(
                    handleTuneable(std::get<I>(tuneablesSubset), std::integral_constant<std::size_t, I>{})...);
            }(std::make_index_sequence<std::tuple_size_v<SubsetTuple>>{});
            return outside_tuple;
            // expand over subset
        }
    };
} // namespace alpaka::tune
#endif // KernelTuningModel
