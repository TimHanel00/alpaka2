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

    using integerType = uint32_t;

    namespace internal
    {


        static integerType getMaxCheckedConfigs_Env()
        {
            if(char const* var = std::getenv("TunerMaxCheckedConfigs")) // all potentially generated
            {
                try
                {
                    integerType value = static_cast<integerType>(std::stoul(var));
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerMaxConfigEvaluations: " << e.what() << std::endl;
                }
            }

            return std::numeric_limits<integerType>::max();
        }

        static integerType getMaxRuns_Env()
        {
            if(char const* var = std::getenv("TunerMaxConfigEvaluations")) // valid configs evaluated
            {
                try
                {
                    integerType value = static_cast<integerType>(std::stoul(var));
                    alpaka::tune::hasMaxRuns_Env(true);
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerMaxConfigEvaluations: " << e.what() << std::endl;
                }
            }

            return std::numeric_limits<integerType>::max();
        }

        static integerType getRunsPerConfig_Env()
        {
            if(char const* var = std::getenv("TunerRunsPerConfig")) // runs per Config
            {
                try
                {
                    integerType const value = static_cast<integerType>(std::stoul(var));
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

        static integerType getMaxConfigs_Env()
        {
            if(char const* var = std::getenv("TunerMaxConfigs")) // max tuning space
            {
                try
                {
                    integerType const value = static_cast<integerType>(std::stoul(var));
                    return value;
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for TunerReRuns: " << e.what() << std::endl;
                }
            }
            return std::numeric_limits<integerType>::max();
        }
    } // namespace internal

    static integerType getMaxCheckConfigs()
    {
        static integerType maxConfigs = internal::getMaxCheckedConfigs_Env();

        return maxConfigs;
    }

    static integerType getMaxConfigs()
    {
        static integerType maxConfigs = internal::getMaxConfigs_Env();

        return maxConfigs;
    }

    static integerType getMaxRuns()
    {
        static integerType maxRuns = internal::getMaxRuns_Env();

        return maxRuns;
    }

    static integerType getRunsPerConfig()
    {
        static integerType maxRuns = internal::getRunsPerConfig_Env();

        return maxRuns;
    }
} // namespace alpaka::tune
#endif // ENVIRONMENTVARS_H
