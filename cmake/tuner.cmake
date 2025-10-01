# Tuner cmake
## load toml11 library
if (NOT DEFINED _TOML11_FETCHED OR NOT _TOML11_FETCHED)
    set(TOML11_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
    message(STATUS "Installing toml11 library that might take a few seconds...")
endif ()
include(FetchContent)

FetchContent_Declare(
        toml11
        GIT_REPOSITORY https://github.com/ToruNiina/toml11.git
        GIT_TAG v4.4.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(toml11)
#target_include_directories(toml11 INTERFACE ${toml11_SOURCE_DIR}/include)

# Create an alias for consistency with the documentation
if (NOT DEFINED _TOML11_FETCHED OR NOT _TOML11_FETCHED)
    message(STATUS "Successfully installed toml11 library")

    set(_TOML11_FETCHED ON CACHE INTERNAL "Flag to avoid fetching toml11 multiple times")
endif ()
# === STRATEGY SELECTION ===
# Allow user to set this from the command line
set(alpaka_Tuner_Strategy "randomExplore" CACHE STRING "Tuning strategy (randomExplore, exhaustive, simulatedAnnealing, randomSample, bayesianOptimization)")
# Strategy selection macro
if (alpaka_Tuner_Strategy STREQUAL "randomExplore")
    set(DTUNER_STRATEGY_MACRO strategy_randomSearch)
elseif (DTuner_Strategy STREQUAL "exhaustive")
    set(DTUNER_STRATEGY_MACRO strategy_exhaustiveSearch)
elseif (DTuner_Strategy STREQUAL "simulatedAnnealing")
    set(DTUNER_STRATEGY_MACRO strategy_simulatedAnnealing)
elseif (DTuner_Strategy STREQUAL "randomSample")
    set(DTUNER_STRATEGY_MACRO strategy_randomSample)
elseif (DTuner_Strategy STREQUAL "bayesianOptimization")
    set(DTUNER_STRATEGY_MACRO strategy_bayesianOptimization)
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
target_link_libraries(alpaka_target_headers INTERFACE toml11::toml11)
target_compile_definitions(alpaka_target_headers INTERFACE ${DTUNER_STRATEGY_MACRO})

