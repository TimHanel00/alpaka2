//
// Created by tim on 18.03.25.
//

#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H

namespace alpaka::tune
{

    inline bool hasRunsPerConfig_Env(std::optional<bool> const& hasRunsPerConfig = std::nullopt)
    {
        static bool hasRunsPerCfg = false;
        if(hasRunsPerConfig.has_value())
        {
            hasRunsPerCfg = hasRunsPerConfig.value();
        }
        return hasRunsPerCfg;
    }

#define upperBoundForRunsPerConfig 50

    inline bool hasMaxRuns_Env(std::optional<bool> const& hasRuns = std::nullopt)
    {
        static bool hasMaxRuns = false;
        if(hasRuns.has_value())
        {
            hasMaxRuns = hasRuns.value();
        }
        return hasMaxRuns;
    }

    namespace internal
    {

        static std::size_t getMaxRuns_Env()
        {
            if(char const* var = std::getenv("TunerMaxConfigEvaluations"))
            {
                try
                {
                    std::size_t value = static_cast<std::size_t>(std::stoul(var));
                    alpaka::tune::hasMaxRuns_Env(true);
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerMaxConfigEvaluations: " << e.what() << std::endl;
                }
            }

            return UINT64_MAX;
        }

        static std::size_t getRunsPerConfig_Env()
        {
            if(char const* var = std::getenv("TunerRunsPerConfig"))
            {
                try
                {
                    std::size_t const value = static_cast<std::size_t>(std::stoul(var));
                    hasRunsPerConfig_Env(true);
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerReRuns: " << e.what() << std::endl;
                }
            }
            return upperBoundForRunsPerConfig;
        }
    #define maxDefaultNumberOfConfigs 32768
        static std::size_t getMaxConfigs_Env()
        {
            if(char const* var = std::getenv("TunerMaxConfigs"))
            {
                try
                {
                    std::size_t const value = static_cast<std::size_t>(std::stoul(var));
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerReRuns: " << e.what() << std::endl;
                }
            }
            return maxDefaultNumberOfConfigs;
        }
    } // namespace internal

    static std::size_t getMaxConfigs()
    {
        static std::size_t maxConfigs = internal::getMaxConfigs_Env();

        return maxConfigs;
    }

    static std::size_t getMaxRuns()
    {
        static std::size_t maxRuns = internal::getMaxRuns_Env();

        return maxRuns;
    }

    static std::size_t getRunsPerConfig()
    {
        static std::size_t maxRuns = internal::getRunsPerConfig_Env();

        return maxRuns;
    }
} // namespace alpaka::tune
#endif // ENVIRONMENTVARS_H
