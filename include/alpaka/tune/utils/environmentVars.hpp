//
// Created by tim on 18.03.25.
//

#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H

static std::size_t getRunsPerConfig_Env()
{
    if(char const* var = std::getenv("TunerRunsPerConfig"))
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
    return 1;
}

static std::size_t getRunsPerConfig(std::optional<std::size_t> const& reRuns = std::nullopt)
{
    static std::size_t envRunsPerConf = getRunsPerConfig_Env();

    if(reRuns.has_value())
    {
        if(reRuns.value() < envRunsPerConf)
        {
            envRunsPerConf = reRuns.value();
        }
    }
    return envRunsPerConf;
}

template<typename T_Context>
static bool userDefMaxRuns(std::optional<bool> userDef = std::nullopt)
{
    static bool isUserDef = false;
    if(userDef.has_value())
    {
        isUserDef = userDef.value();
    }

    return isUserDef;
}

template<typename T_Context>
static std::size_t getMaxRuns_Env()
{
    if(char const* var = std::getenv("TunerMaxConfigs"))
    {
        try
        {
            std::size_t value = static_cast<std::size_t>(std::stoul(var));
            userDefMaxRuns<T_Context>(std::make_optional<bool>(true));
            return value;
        }
        catch(std::exception const& e)
        {
            std::cerr << "Invalid value for TunerMaxConfigs: " << e.what() << std::endl;
        }
    }

    return UINT64_MAX;
}

template<typename T_Context>
inline std::size_t getMaxRuns(std::optional<std::size_t> const& maxRuns = std::nullopt)
{
    static std::size_t envMaxRuns = getMaxRuns_Env<T_Context>();

    if(maxRuns.has_value())
    {
        if(maxRuns.value() < envMaxRuns)
        {
            envMaxRuns = maxRuns.value();
        }
    }
    return envMaxRuns; // return maximal achievable value
}
#endif // ENVIRONMENTVARS_H
