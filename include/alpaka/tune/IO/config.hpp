//
// Created by tim on 13.10.25.
//

#ifndef CONFIG_H
#define CONFIG_H
#include <alpaka/tune/concepts.hpp>

#include <cmath>

template<alpaka::tune::concepts::Integral T, uint32_t NumTunables>
struct Config
{
    using value_type = T;
    std::array<value_type, NumTunables> config{};

    explicit Config(std::array<value_type, NumTunables> const& config) : config(config)
    {
        static_assert(NumTunables > 0, " A parameter configuration can not be empty!");
    };

    bool operator==(Config const& other) const noexcept
    {
        return config == other.config;
    }
};

template<alpaka::tune::concepts::Floating T, uint32_t NumTunables>
struct NormalizedConfig
{
    using value_type = T;
    std::array<value_type, NumTunables> config{};

    explicit NormalizedConfig(std::array<value_type, NumTunables> const& config) : config(config)
    {
        static_assert(NumTunables > 0, " A parameter configuration can not be empty!");
        for(auto const& val : config)
        {
            assert(0.0 <= val && val <= 1.0, "normalized Config values have to be between 0.0 and 1.0!");
        }
    };

    bool operator==(NormalizedConfig const& other) const noexcept
    {
        // compare with tolerance for floating point
        constexpr double eps = 1e-9;
        for(std::size_t i = 0; i < NumTunables; ++i)
            if(std::fabs(config[i] - other.config[i]) > eps)
                return false;
        return true;
    }
};

namespace std
{
    template<alpaka::tune::concepts::Integral T, auto N>
    struct hash<Config<T, N>>
    {
        std::size_t operator()(Config<T, N> const& c) const noexcept
        {
            std::size_t seed = 0;
            std::hash<T> hasher;
            for(auto const& v : c.eonfig)
            {
                // combine hash
                seed ^= hasher(v) + 0x9e37'79b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
} // namespace std

template<typename T, std::size_t N>
Config(std::array<T, N>) -> Config<T, N>;
#endif // CONFIG_H
