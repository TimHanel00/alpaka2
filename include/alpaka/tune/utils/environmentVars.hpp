//
// Created by tim on 18.03.25.
//

#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H

inline bool hasRunsPerConfig_Env(std::optional<bool> const& hasRunsPerConfig = std::nullopt)
{
    static bool hasRunsPerCfg = false;
    if(hasRunsPerConfig.has_value())
    {
        hasRunsPerCfg = hasRunsPerConfig.value();
    }
    return hasRunsPerCfg;
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
    return 1;
}

inline bool hasMaxRuns_Env(std::optional<bool> const& hasRuns = std::nullopt)
{
    static bool hasMaxRuns = false;
    if(hasRuns.has_value())
    {
        hasMaxRuns = hasRuns.value();
    }
    return hasMaxRuns;
}

static std::size_t getMaxRuns_Env()
{
    if(char const* var = std::getenv("TunerMaxConfigEvaluations"))
    {
        try
        {
            std::size_t value = static_cast<std::size_t>(std::stoul(var));
            hasMaxRuns_Env(true);
            return value;
        }
        catch(std::exception const& e)
        {
            std::cerr << "Invalid value for TunerMaxConfigEvaluations: " << e.what() << std::endl;
        }
    }

    return UINT64_MAX;
}
#endif // ENVIRONMENTVARS_H
