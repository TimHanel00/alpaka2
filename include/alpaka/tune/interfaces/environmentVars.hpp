//
// Created by tim on 18.03.25.
//

#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H

namespace alpaka::tune
{
    // =====================================================================================
    // Your existing flags stay exactly as-is (interface preserved)
    // =====================================================================================
    inline bool hasRunsPerConfig_Env(std::optional<bool> const& hasRunsPerConfig = std::nullopt)
    {
        static bool hasRunsPerCfg = false;
        if(hasRunsPerConfig.has_value())
            hasRunsPerCfg = hasRunsPerConfig.value();
        return hasRunsPerCfg;
    }

#ifndef upperBoundForRunsPerConfig
#    define upperBoundForRunsPerConfig 50
#endif

    inline bool hasMaxRuns_Env(std::optional<bool> const& hasRuns = std::nullopt)
    {
        static bool hasMaxRuns = false;
        if(hasRuns.has_value())
            hasMaxRuns = hasRuns.value();
        return hasMaxRuns;
    }

    // =====================================================================================
    // Setters live in alpaka::tune::Vars (new), with env precedence
    // =====================================================================================
    namespace Vars
    {

        using integerType = std::uint32_t;

        namespace detail
        {
            struct State
            {
                // values
                integerType maxCheckedConfigs = std::numeric_limits<integerType>::max();
                integerType maxConfigs = std::numeric_limits<integerType>::max();
                integerType maxRuns = std::numeric_limits<integerType>::max();
                integerType runsPerConfig = static_cast<integerType>(upperBoundForRunsPerConfig);

                // whether env provided a value (locks the field against setters)
                bool envMaxCheckedConfigs = false;
                bool envMaxConfigs = false;
                bool envMaxRuns = false;
                bool envRunsPerConfig = false;

                bool inited = false;
            };

            inline State& st()
            {
                static State s;
                return s;
            }

            inline std::optional<integerType> parseEnvU32(char const* name)
            {
                if(char const* var = std::getenv(name))
                {
                    try
                    {
                        return static_cast<integerType>(std::stoul(var));
                    }
                    catch(std::exception const& e)
                    {
                        std::cerr << "Invalid value for " << name << ": " << e.what() << '\n';
                    }
                }
                return std::nullopt;
            }

            inline void initFromEnvOnce()
            {
                auto& s = st();
                if(s.inited)
                    return;
                s.inited = true;

                if(auto v = parseEnvU32("TunerMaxCheckedConfigs"))
                {
                    s.maxCheckedConfigs = *v;
                    s.envMaxCheckedConfigs = true;
                }
                if(auto v = parseEnvU32("TunerMaxConfigs"))
                {
                    s.maxConfigs = *v;
                    s.envMaxConfigs = true;
                }
                if(auto v = parseEnvU32("TunerMaxConfigEvaluations"))
                { // valid configs evaluated
                    s.maxRuns = *v;
                    s.envMaxRuns = true;
                    // preserve original side-effect
                    alpaka::tune::hasMaxRuns_Env(true);
                }
                if(auto v = parseEnvU32("TunerRunsPerConfig"))
                { // runs per config
                    s.runsPerConfig = *v;
                    s.envRunsPerConfig = true;
                    // preserve original side-effect
                    alpaka::tune::hasRunsPerConfig_Env(true);
                }
            }
        } // namespace detail

        // ---------------- Getters (Vars) ----------------
        inline std::uint32_t getMaxCheckedConfigs()
        {
            detail::initFromEnvOnce();
            return detail::st().maxCheckedConfigs;
        }

        inline std::uint32_t getMaxConfigs()
        {
            detail::initFromEnvOnce();
            return detail::st().maxConfigs;
        }

        inline std::uint32_t getMaxRuns()
        {
            detail::initFromEnvOnce();
            return detail::st().maxRuns;
        }

        inline std::uint32_t getRunsPerConfig()
        {
            detail::initFromEnvOnce();
            return detail::st().runsPerConfig;
        }

        // ---------------- Setters (Vars) ----------------
        // Return true if the set took effect; false if an env var is present (env wins).

        inline bool setMaxCheckedConfigs(std::uint32_t v)
        {
            detail::initFromEnvOnce();
            auto& s = detail::st();
            if(s.envMaxCheckedConfigs)
                return false;
            s.maxCheckedConfigs = v;
            return true;
        }

        inline bool setMaxConfigs(std::uint32_t v)
        {
            detail::initFromEnvOnce();
            auto& s = detail::st();
            if(s.envMaxConfigs)
                return false;
            s.maxConfigs = v;
            return true;
        }

        inline bool setMaxRuns(std::uint32_t v)
        {
            detail::initFromEnvOnce();
            auto& s = detail::st();
            if(s.envMaxRuns)
                return false;
            s.maxRuns = v;
            // mirror original flag semantics
            alpaka::tune::hasMaxRuns_Env(true);
            return true;
        }

        inline bool setRunsPerConfig(std::uint32_t v)
        {
            detail::initFromEnvOnce();
            auto& s = detail::st();
            if(s.envRunsPerConfig)
                return false;
            s.runsPerConfig = v;
            // mirror original flag semantics
            alpaka::tune::hasRunsPerConfig_Env(true);
            return true;
        }

    } // namespace Vars

    // =====================================================================================
    // Your original public getters remain in alpaka::tune (same names/signatures).
    // They now forward to Vars getters (no static locals), so setters can take effect,
    // while env variables still override via Vars::detail.
    // =====================================================================================
    using Vars::integerType; // keep your integerType visible if you relied on it

    static integerType getMaxCheckConfigs()
    {
        return Vars::getMaxCheckedConfigs();
    }

    static integerType getMaxConfigs()
    {
        return Vars::getMaxConfigs();
    }

    static integerType getMaxRuns()
    {
        return Vars::getMaxRuns();
    }

    static integerType getRunsPerConfig()
    {
        return Vars::getRunsPerConfig();
    }
} // namespace alpaka::tune
#endif // ENVIRONMENTVARS_H
