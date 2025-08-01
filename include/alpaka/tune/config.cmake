# strategyConfig.cmake

# === STRATEGY SELECTION ===
# Allow user to set this from the command line
set(DTuner_Strategy "randomExplore" CACHE STRING "Tuning strategy (randomExplore, exhaustive, simulatedAnnealing, randomSample, bayesianOptimization)")
set_property(CACHE DTuner_Strategy PROPERTY STRINGS randomExplore exhaustive simulatedAnnealing randomSample bayesianOptimization)

# Strategy selection macro
if (DTuner_Strategy STREQUAL "randomExplore")
    set(DTUNER_STRATEGY_MACRO strategy_randomSearch)
elseif (DTuner_Strategy STREQUAL "exhaustive")
    set(DTUNER_STRATEGY_MACRO strategy_exhaustiveSearch)
elseif (DTuner_Strategy STREQUAL "simulatedAnnealing")
    set(DTUNER_STRATEGY_MACRO strategy_simulatedAnnealing)
elseif (DTuner_Strategy STREQUAL "randomSample")
    set(DTUNER_STRATEGY_MACRO strategy_randomSample)
elseif (DTuner_Strategy STREQUAL "bayesianOptimization")
    set(DTUNER_STRATEGY_MACRO strategy_bayesianOptimization)

    include(FetchContent)
    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
else ()
    message(FATAL_ERROR "Invalid DTuner_Strategy: ${DTuner_Strategy}")
endif ()

